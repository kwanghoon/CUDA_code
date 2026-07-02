// pitfalls_bench.cu
// 흔한 실수 갤러리: 안티패턴(BAD) vs 정답(GOOD)을 쌍으로 실측해 느림을 증명.
#include "../common/raii.cuh"

#include <cstdio>
#include <string>
#include <vector>

constexpr int BLK = 256;

// 1) Coalescing: 연속 vs 32-strided 접근
__global__ void copyCoalesced(const int* __restrict__ in, int* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i];                        // 연속 → 128B 트랜잭션 1회
}
__global__ void copyStrided(const int* __restrict__ in, int* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[(i * 32) % n];             // BAD: 32칸 점프 → 트랜잭션 32배
}

// 2) Reduction atomic: 전역 원소별 vs shared 트리
__global__ void reduceAtomicBad(const int* __restrict__ in, int* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(out, in[i]);                 // BAD: 단일 카운터에 전역 atomic 경합
}
__global__ void reduceSharedGood(const int* __restrict__ in, int* __restrict__ out, int n) {
    __shared__ int s[BLK];
    int tid = threadIdx.x, i = blockIdx.x * blockDim.x + tid;
    s[tid] = (i < n) ? in[i] : 0; __syncthreads();
    for (int st = BLK / 2; st > 0; st >>= 1) { if (tid < st) s[tid] += s[tid + st]; __syncthreads(); }
    if (tid == 0) atomicAdd(out, s[0]);               // 블록당 atomic 1회
}

// 3) Warp divergence: 워프 내 홀짝 분기 vs 워프 정렬 분기
__global__ void divergentBad(const int* __restrict__ in, int* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int reps = (threadIdx.x & 1) ? 512 : 4;   // BAD: 워프 내 반복수 갈림 → 워프가 512까지 마스크 실행
    int v = in[i];
    for (int k = 0; k < reps; ++k) v = v * 3 + 1;
    out[i] = v;
}
__global__ void convergentGood(const int* __restrict__ in, int* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int reps = ((threadIdx.x >> 5) & 1) ? 512 : 4;   // 워프 단위 정렬(워프별 512 또는 4)
    int v = in[i];
    for (int k = 0; k < reps; ++k) v = v * 3 + 1;
    out[i] = v;
}

// 4) Register spill: 큰 지역 배열 vs 스칼라
// buf[64] 동적 인덱싱 → 레지스터 승격 불가 → local memory 스필 (nvcc -Xptxas -v 로 확인).
__global__ void spillBad(const int* __restrict__ in, int* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int buf[64];
    #pragma unroll
    for (int k = 0; k < 64; ++k) buf[k] = in[(i + k) % n];
    int s = 0;
    for (int k = 0; k < 64; ++k) s += buf[(s + k) & 63];   // 동적 인덱싱 → buf가 local memory
    out[i] = s;
}
__global__ void spillGood(const int* __restrict__ in, int* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int s = 0;
    #pragma unroll
    for (int k = 0; k < 64; ++k) s += in[(i + k) % n];     // 스칼라 누적 → 레지스터, 스필 없음
    out[i] = s;
}

// 5) Bank conflict: shared 전치 읽기, 패딩 없음 vs +1
__global__ void bankBad(const int* __restrict__ in, int* __restrict__ out, int n) {
    __shared__ int s[32][32];                 // BAD: 패딩 없음
    int t = threadIdx.x, base = blockIdx.x * 1024, row = t >> 5, col = t & 31;
    s[row][col] = in[base + t]; __syncthreads();
    out[base + t] = s[col][row];              // 전치 읽기 → 32-way bank conflict
}
__global__ void bankGood(const int* __restrict__ in, int* __restrict__ out, int n) {
    __shared__ int s[32][33];                 // +1 패딩 → 뱅크 분산
    int t = threadIdx.x, base = blockIdx.x * 1024, row = t >> 5, col = t & 31;
    s[row][col] = in[base + t]; __syncthreads();
    out[base + t] = s[col][row];
}

// 6) Small grid: 그리드가 작아 SM 놀림 (같은 grid-stride 커널, 런치만 다름)
__global__ void copyGS(const int* __restrict__ in, int* __restrict__ out, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        out[i] = in[i] + 1;
}

int main() {
    const int n = 1 << 24;
    std::vector<int> h(n, 1);
    DeviceBuffer<int> d_in(n), d_out(n), d_scalar(1);
    d_in.copyFromHost(h.data());

    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    std::printf("=== 흔한 실수 갤러리 (N=%d, Orin SM %d.%d) ===\n", n, prop.major, prop.minor);
    std::printf("%-28s %10s %10s\n", "구현", "ms", "느림배율");
    std::printf("%s\n", std::string(50, '-').c_str());

    int grid = (n + BLK - 1) / BLK, iters = 50;
    auto timeit = [&](auto launch) {
        launch(); CHECK_CUDA(cudaDeviceSynchronize());
        GpuTimer t; t.start(); for (int i = 0; i < iters; ++i) launch();
        return t.stop() / iters;
    };
    auto pair = [&](const char* title, auto bad, auto good) {
        double tg = timeit(good), tb = timeit(bad);
        double r = tb / tg;
        const char* note = (r > 1.1) ? "  <-- 느림(안티패턴)"
                         : (r < 0.9) ? "  (Orin에선 오히려 빠름 - HW 발전)"
                                     : "  (차이 미미)";
        std::printf("%-28s %10.4f %9.2fx\n", (std::string(title) + " GOOD").c_str(), tg, 1.0);
        std::printf("%-28s %10.4f %9.2fx%s\n", (std::string(title) + " BAD").c_str(), tb, r, note);
    };

    pair("1.coalescing",
         [&] { copyStrided<<<grid, BLK>>>(d_in.data(), d_out.data(), n); },
         [&] { copyCoalesced<<<grid, BLK>>>(d_in.data(), d_out.data(), n); });
    pair("2.reduction atomic",
         [&] { cudaMemset(d_scalar.data(), 0, 4); reduceAtomicBad<<<grid, BLK>>>(d_in.data(), d_scalar.data(), n); },
         [&] { cudaMemset(d_scalar.data(), 0, 4); reduceSharedGood<<<grid, BLK>>>(d_in.data(), d_scalar.data(), n); });
    pair("3.warp divergence",
         [&] { divergentBad<<<grid, BLK>>>(d_in.data(), d_out.data(), n); },
         [&] { convergentGood<<<grid, BLK>>>(d_in.data(), d_out.data(), n); });
    pair("4.register spill",
         [&] { spillBad<<<grid, BLK>>>(d_in.data(), d_out.data(), n); },
         [&] { spillGood<<<grid, BLK>>>(d_in.data(), d_out.data(), n); });
    pair("5.bank conflict",
         [&] { bankBad<<<n / 1024, 1024>>>(d_in.data(), d_out.data(), n); },
         [&] { bankGood<<<n / 1024, 1024>>>(d_in.data(), d_out.data(), n); });
    int smGrid = prop.multiProcessorCount * 32;
    pair("6.small grid (SM 놀림)",
         [&] { copyGS<<<8, BLK>>>(d_in.data(), d_out.data(), n); },
         [&] { copyGS<<<smGrid, BLK>>>(d_in.data(), d_out.data(), n); });
    return 0;
}
