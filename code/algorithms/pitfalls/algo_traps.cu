// algo_traps.cu
// 알고리즘별 L0(naive) 정확성 함정: BUG vs GOOD 쌍으로 결과가 틀림(Check=FAIL)을 보임.
//   compute-sanitizer --tool racecheck ./algo_traps 로 레이스도 검출.
#include "../common/raii.cuh"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

static void row(const char* algo, const char* variant, bool ok, const char* note) {
    std::printf("%-14s %-26s %-7s %s\n", algo, variant, ok ? "OK" : "FAIL", note);
}

// reduction: 리덕션 단계 사이 __syncthreads 누락
__global__ void redGood(const int* in, int* out, int n) {
    __shared__ int s[256];
    int t = threadIdx.x, i = blockIdx.x * 256 + t;
    s[t] = (i < n) ? in[i] : 0;
    __syncthreads();
    for (int st = 128; st > 0; st >>= 1) { if (t < st) s[t] += s[t + st]; __syncthreads(); }
    if (t == 0) atomicAdd(out, s[0]);
}
__global__ void redNoSync(const int* in, int* out, int n) {
    __shared__ int s[256];
    int t = threadIdx.x, i = blockIdx.x * 256 + t;
    s[t] = (i < n) ? in[i] : 0;                       // BUG: __syncthreads() 누락
    for (int st = 128; st > 0; st >>= 1) { if (t < st) s[t] += s[t + st]; /* BUG: sync 누락 */ }
    if (t == 0) atomicAdd(out, s[0]);                 // 워프 간 미동기화 → 부분합 유실(비결정)
}

// histogram: 원자연산 없이 증가 → 갱신 유실
__global__ void histGood(const int* in, int* h, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        atomicAdd(&h[in[i] & 255], 1);
}
__global__ void histNoAtomic(const int* in, int* h, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        h[in[i] & 255]++;                             // BUG: read-modify-write 레이스 → 카운트 유실
}

// scan(Hillis-Steele): in-place 단일 버퍼 → RAW 해저드. sync가 있어도 이웃이 갱신돼 오염 → 더블버퍼 필요.
constexpr int SCAN_N = 1024;
__global__ void scanGood(const int* in, int* out, int n) {
    __shared__ int b[SCAN_N];
    int t = threadIdx.x;
    b[t] = (t < n) ? in[t] : 0;
    __syncthreads();
    for (int off = 1; off < SCAN_N; off <<= 1) {
        int v = (t >= off) ? b[t] + b[t - off] : b[t];   // 먼저 레지스터로 읽고
        __syncthreads();
        b[t] = v;                                        // 그다음 씀 (RAW 회피)
        __syncthreads();
    }
    if (t < n) out[t] = b[t];
}
__global__ void scanInPlaceBad(const int* in, int* out, int n) {
    __shared__ int b[SCAN_N];
    int t = threadIdx.x;
    b[t] = (t < n) ? in[t] : 0;
    __syncthreads();
    for (int off = 1; off < SCAN_N; off <<= 1) {
        if (t >= off) b[t] += b[t - off];   // BUG: in-place — 이웃 b[t-off]가 이 step서 갱신 중(cross-warp RAW race)
        __syncthreads();
    }
    if (t < n) out[t] = b[t];
}

// matmul(tiled): shared 타일 로드 후 __syncthreads 누락
constexpr int TZ = 16;
__global__ void mmGood(const float* A, const float* B, float* C, int N) {
    __shared__ float As[TZ][TZ], Bs[TZ][TZ];
    int r = blockIdx.y * TZ + threadIdx.y, c = blockIdx.x * TZ + threadIdx.x;
    float s = 0;
    for (int t = 0; t < N; t += TZ) {
        As[threadIdx.y][threadIdx.x] = A[r * N + t + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(t + threadIdx.y) * N + c];
        __syncthreads();
        for (int k = 0; k < TZ; ++k) s += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    C[r * N + c] = s;
}
__global__ void mmNoSync(const float* A, const float* B, float* C, int N) {
    __shared__ float As[TZ][TZ], Bs[TZ][TZ];
    int r = blockIdx.y * TZ + threadIdx.y, c = blockIdx.x * TZ + threadIdx.x;
    float s = 0;
    for (int t = 0; t < N; t += TZ) {
        As[threadIdx.y][threadIdx.x] = A[r * N + t + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(t + threadIdx.y) * N + c];
        // BUG: 로드 후 __syncthreads() 누락 → 남의 타일이 안 채워진 채로 곱함
        for (int k = 0; k < TZ; ++k) s += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        // BUG: 다음 타일 로드 전 __syncthreads() 누락 → 아직 쓰는 중인 타일을 덮어씀
    }
    C[r * N + c] = s;
}

// softmax: 최댓값 빼기 생략 → exp 오버플로
__global__ void smGood(const float* x, float* y, int n) {
    float m = -1e30f; for (int i = 0; i < n; ++i) m = fmaxf(m, x[i]);
    float s = 0;      for (int i = 0; i < n; ++i) s += expf(x[i] - m);
    for (int i = 0; i < n; ++i) y[i] = expf(x[i] - m) / s;
}
__global__ void smNoMax(const float* x, float* y, int n) {
    float s = 0; for (int i = 0; i < n; ++i) s += expf(x[i]);   // BUG: max 안 뺌 → 큰 값서 exp=Inf
    for (int i = 0; i < n; ++i) y[i] = expf(x[i]) / s;          // Inf/Inf = NaN
}

// convolution(1D): 경계 클램프 생략 → 배열 밖 접근/잘못된 가장자리
__global__ void convGood(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float s = 0;
    for (int d = -2; d <= 2; ++d) { int j = min(max(i + d, 0), n - 1); s += in[j]; }  // 클램프
    out[i] = s / 5.0f;
}
__global__ void convNoClamp(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float s = 0;
    for (int d = -2; d <= 2; ++d) s += in[i + d];              // BUG: i±2 경계 밖 접근(가장자리 오염/OOB)
    out[i] = s / 5.0f;
}

int main() {
    const int n = 1 << 20;
    std::printf("=== 알고리즘별 베이스라인 흔한 실수 (N=%d) ===\n", n);
    std::printf("%-14s %-26s %-7s %s\n", "알고리즘", "변형", "Check", "실수 포인트");
    std::printf("%s\n", std::string(78, '-').c_str());

    // reduction
    {
        std::vector<int> h(n, 1); long cpu = n;
        DeviceBuffer<int> din(n), d(1);
        din.copyFromHost(h.data());
        int grid = (n + 255) / 256, zero = 0, r = 0;
        auto run = [&](void (*k)(const int*, int*, int)) {
            d.copyFromHost(&zero); k<<<grid, 256>>>(din.data(), d.data(), n);
            CHECK_CUDA(cudaDeviceSynchronize()); d.copyToHost(&r); return (long)r;
        };
        row("reduction", "good (__syncthreads)", run(redGood) == cpu, "");
        row("reduction", "BUG: sync 누락", run(redNoSync) == cpu, "리덕션 단계 사이 배리어 필요 (racecheck)");
    }
    // histogram
    {
        std::vector<int> h(n); for (int i = 0; i < n; ++i) h[i] = i & 255;
        DeviceBuffer<int> din(n), dh(256);
        din.copyFromHost(h.data());
        std::vector<int> zero(256, 0), out(256);
        auto total = [&](void (*k)(const int*, int*, int)) {
            dh.copyFromHost(zero.data()); k<<<256, 256>>>(din.data(), dh.data(), n);
            CHECK_CUDA(cudaDeviceSynchronize()); dh.copyToHost(out.data());
            long t = 0; for (int c = 0; c < 256; ++c) t += out[c]; return t;
        };
        row("histogram", "good (atomicAdd)", total(histGood) == n, "");
        row("histogram", "BUG: atomic 누락", total(histNoAtomic) == n, "동시 증가 유실 → 합 < N");
    }
    // scan (블록 1개, SCAN_N 원소 inclusive scan)
    {
        int m = SCAN_N; std::vector<int> x(m), ref(m); int acc = 0;
        for (int i = 0; i < m; ++i) { x[i] = (i % 7) - 3; acc += x[i]; ref[i] = acc; }
        DeviceBuffer<int> dx(m), dy(m); dx.copyFromHost(x.data());
        std::vector<int> y(m);
        auto ok = [&](void (*k)(const int*, int*, int)) {
            k<<<1, m>>>(dx.data(), dy.data(), m); CHECK_CUDA(cudaDeviceSynchronize()); dy.copyToHost(y.data());
            for (int i = 0; i < m; ++i) if (y[i] != ref[i]) return false; return true;
        };
        row("scan", "good (더블버퍼/레지스터)", ok(scanGood), "");
        row("scan", "BUG: in-place RAW", ok(scanInPlaceBad), "in-place 이웃 갱신 중 읽음 → 더블버퍼 필요(race,racecheck)");
    }
    // matmul
    {
        int N = 256; std::vector<float> A(N * N), B(N * N), ref(N * N, 0), out(N * N);
        for (int i = 0; i < N * N; ++i) { A[i] = (i % 7) * 0.1f; B[i] = (i % 5) * 0.1f; }
        for (int i = 0; i < N; ++i) for (int j = 0; j < N; ++j) { float s = 0; for (int k = 0; k < N; ++k) s += A[i * N + k] * B[k * N + j]; ref[i * N + j] = s; }
        DeviceBuffer<float> dA(N * N), dB(N * N), dC(N * N);
        dA.copyFromHost(A.data()); dB.copyFromHost(B.data());
        dim3 b(TZ, TZ), g(N / TZ, N / TZ);
        auto ok = [&](void (*k)(const float*, const float*, float*, int)) {
            k<<<g, b>>>(dA.data(), dB.data(), dC.data(), N); CHECK_CUDA(cudaDeviceSynchronize());
            dC.copyToHost(out.data());
            for (int i = 0; i < N * N; ++i) if (std::fabs(out[i] - ref[i]) > 1e-2f * (1 + std::fabs(ref[i]))) return false;
            return true;
        };
        row("matmul", "good (__syncthreads)", ok(mmGood), "");
        row("matmul", "BUG: 타일 sync 누락", ok(mmNoSync), "shared 타일 로드 전후 배리어 필요");
    }
    // softmax
    {
        int m = 2048; std::vector<float> x(m); for (int i = 0; i < m; ++i) x[i] = 60.0f + (i % 40); // 큰 값
        DeviceBuffer<float> dx(m), dy(m); dx.copyFromHost(x.data());
        std::vector<float> y(m);
        auto finite_sum1 = [&](void (*k)(const float*, float*, int)) {
            k<<<1, 1>>>(dx.data(), dy.data(), m); CHECK_CUDA(cudaDeviceSynchronize()); dy.copyToHost(y.data());
            double s = 0; bool fin = true; for (int i = 0; i < m; ++i) { if (!std::isfinite(y[i])) fin = false; s += y[i]; }
            return fin && std::fabs(s - 1.0) < 1e-2;   // 정상 softmax는 합=1, 유한
        };
        row("softmax", "good (max 빼기)", finite_sum1(smGood), "");
        row("softmax", "BUG: max 안 뺌", finite_sum1(smNoMax), "exp(큰 값)=Inf → NaN (수치 불안정)");
    }
    // convolution
    {
        std::vector<float> in(n, 1.0f), ref(n), out(n);
        for (int i = 0; i < n; ++i) { float s = 0; for (int d = -2; d <= 2; ++d) { int j = std::min(std::max(i + d, 0), n - 1); s += in[j]; } ref[i] = s / 5.0f; }
        DeviceBuffer<float> din(n), dout(n); din.copyFromHost(in.data());
        int grid = (n + 255) / 256;
        auto ok = [&](void (*k)(const float*, float*, int)) {
            k<<<grid, 256>>>(din.data(), dout.data(), n); CHECK_CUDA(cudaDeviceSynchronize()); dout.copyToHost(out.data());
            for (int i = 0; i < n; ++i) if (std::fabs(out[i] - ref[i]) > 1e-3f) return false;
            return true;
        };
        row("convolution", "good (경계 클램프)", ok(convGood), "");
        row("convolution", "BUG: 클램프 누락", ok(convNoClamp), "가장자리서 배열 밖 접근 (OOB/오염)");
    }

    std::printf("\n각 BUG는 해당 알고리즘 L0(naive)에서 초보가 흔히 빠지는 함정이다.\n");
    std::printf("주의: reduction/histogram/matmul/softmax/conv 는 결과가 바로 틀린다(loud).\n");
    std::printf("      scan in-place 는 '조용한(silent) 레이스' — 여기선 우연히 OK지만 실제 버그다\n");
    std::printf("      → compute-sanitizer --tool racecheck ./algo_traps 로 hazard 검출(make traps_racecheck).\n");
    return 0;
}
