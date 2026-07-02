// 실습: N-body (중력) 최적화 — SOLUTION (누적 케이스 래더 L0→L3, 참고답안)
// baseline(L0 AoS)에서 한 기법씩 쌓아 올린 여러 케이스. 각 레벨 = 이전 + 기법 1개.
//   L0 AoS naive         : struct Body[] , 타일링 없음(전역 반복 로드)
//   L1 SoA               : x[],y[],z[],m[] 분리 배열 (자기-body 읽기 coalesced)
//   L2 SoA + shared tile : body 타일을 shared로 올려 재사용 (전역 O(N²) → O(N²/blockDim))
//   L3 + unroll/fast-math : 내부 루프 #pragma unroll (compute 파이프 채움)
//   L0→L1  gld 효율/sector 활용 (AoS strided → SoA coalesced)
//   L1→L2  dram__bytes.sum (타일 재사용 → 전역 트래픽↓)
//   L2→L3  sm__throughput / fp32 (unroll로 FMA 파이프↑)
// 참고: 이 커널은 b[j] broadcast + rsqrtf 많아 처음부터 compute-bound라 Orin에선 L3에서 주로 오른다.
#include "../../common/raii.cuh"
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>

struct Body { float x, y, z, m; };

// L0: AoS, 타일링 없음
__global__ void nbodyL0_AoS(const Body* __restrict__ b, float3* __restrict__ acc, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    float xi = b[i].x, yi = b[i].y, zi = b[i].z, ax = 0, ay = 0, az = 0;
    for (int j = 0; j < n; ++j) {
        float dx = b[j].x - xi, dy = b[j].y - yi, dz = b[j].z - zi;
        float inv = rsqrtf(dx * dx + dy * dy + dz * dz + 1e-4f), s = b[j].m * inv * inv * inv;
        ax += dx * s; ay += dy * s; az += dz * s;
    }
    acc[i] = make_float3(ax, ay, az);
}
// L1: SoA (레이아웃만 교체), 타일링 없음
__global__ void nbodyL1_SoA(const float* __restrict__ x, const float* __restrict__ y,
                            const float* __restrict__ z, const float* __restrict__ m,
                            float3* __restrict__ acc, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    float xi = x[i], yi = y[i], zi = z[i], ax = 0, ay = 0, az = 0;
    for (int j = 0; j < n; ++j) {
        float dx = x[j] - xi, dy = y[j] - yi, dz = z[j] - zi;
        float inv = rsqrtf(dx * dx + dy * dy + dz * dz + 1e-4f), s = m[j] * inv * inv * inv;
        ax += dx * s; ay += dy * s; az += dz * s;
    }
    acc[i] = make_float3(ax, ay, az);
}
// L2/L3: SoA + shared 타일. Unroll=true 면 내부 루프 언롤.
template <bool Unroll>
__global__ void nbodySoATiled(const float* __restrict__ x, const float* __restrict__ y,
                              const float* __restrict__ z, const float* __restrict__ m,
                              float3* __restrict__ acc, int n) {
    extern __shared__ float4 tile[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float xi = (i < n) ? x[i] : 0, yi = (i < n) ? y[i] : 0, zi = (i < n) ? z[i] : 0, ax = 0, ay = 0, az = 0;
    for (int t = 0; t < n; t += blockDim.x) {
        int j = t + threadIdx.x;
        tile[threadIdx.x] = make_float4(j < n ? x[j] : 0, j < n ? y[j] : 0, j < n ? z[j] : 0, j < n ? m[j] : 0);
        __syncthreads();
        if constexpr (Unroll) {
            #pragma unroll 8
            for (int k = 0; k < blockDim.x && t + k < n; ++k) {
                float4 bj = tile[k]; float dx = bj.x - xi, dy = bj.y - yi, dz = bj.z - zi;
                float inv = rsqrtf(dx * dx + dy * dy + dz * dz + 1e-4f), s = bj.w * inv * inv * inv;
                ax += dx * s; ay += dy * s; az += dz * s;
            }
        } else {
            for (int k = 0; k < blockDim.x && t + k < n; ++k) {
                float4 bj = tile[k]; float dx = bj.x - xi, dy = bj.y - yi, dz = bj.z - zi;
                float inv = rsqrtf(dx * dx + dy * dy + dz * dz + 1e-4f), s = bj.w * inv * inv * inv;
                ax += dx * s; ay += dy * s; az += dz * s;
            }
        }
        __syncthreads();
    }
    if (i < n) acc[i] = make_float3(ax, ay, az);
}

int main(int argc, char** argv) {
    int n = (argc > 1) ? atoi(argv[1]) : (1 << 14);
    std::vector<float> hx(n), hy(n), hz(n), hm(n); std::vector<Body> hb(n);
    unsigned s = 7u;
    for (int i = 0; i < n; ++i) {
        s = s * 1103515245u + 12345u; hx[i] = ((s >> 16) & 1023) * 0.1f;
        s = s * 1103515245u + 12345u; hy[i] = ((s >> 16) & 1023) * 0.1f;
        s = s * 1103515245u + 12345u; hz[i] = ((s >> 16) & 1023) * 0.1f;
        hm[i] = 1.0f; hb[i] = {hx[i], hy[i], hz[i], 1.0f};
    }
    DeviceBuffer<Body> db(n); DeviceBuffer<float> dx(n), dy(n), dz(n), dm(n); DeviceBuffer<float3> dacc(n);
    db.copyFromHost(hb.data());
    dx.copyFromHost(hx.data()); dy.copyFromHost(hy.data()); dz.copyFromHost(hz.data()); dm.copyFromHost(hm.data());

    int block = 256, grid = (n + block - 1) / block; size_t smem = block * sizeof(float4);
    double rax = 0, ray = 0, raz = 0;
    for (int j = 0; j < n; ++j) { double dxj = hx[j] - hx[0], dyj = hy[j] - hy[0], dzj = hz[j] - hz[0];
        double inv = 1.0 / std::sqrt(dxj * dxj + dyj * dyj + dzj * dzj + 1e-4), sf = hm[j] * inv * inv * inv;
        rax += dxj * sf; ray += dyj * sf; raz += dzj * sf; }

    std::printf("=== N-body 누적 케이스 (N=%d) ===\n", n);
    std::printf("%-26s %10s %12s %8s %10s\n", "Level", "ms", "GFLOP/s", "Check", "Speedup");
    std::printf("%s\n", std::string(70, '-').c_str());
    double base = -1.0;
    std::vector<float3> acc(n);
    auto bench = [&](const char* name, auto launch) {
        launch(); CHECK_CUDA(cudaDeviceSynchronize()); dacc.copyToHost(acc.data());
        bool ok = std::fabs(acc[0].x - rax) < 1e-1 * (1 + std::fabs(rax));
        GpuTimer t; t.start(); const int it = 10; for (int k = 0; k < it; ++k) launch();
        double ms = t.stop() / it; if (base < 0) base = ms;
        std::printf("%-26s %10.4f %12.1f %8s %9.2fx\n", name, ms, 20.0 * (double)n * n / (ms / 1e3) / 1e9,
                    ok ? "OK" : "FAIL", base / ms);
    };
    bench("L0 AoS naive", [&] { nbodyL0_AoS<<<grid, block>>>(db.data(), dacc.data(), n); });
    bench("L1 SoA", [&] { nbodyL1_SoA<<<grid, block>>>(dx.data(), dy.data(), dz.data(), dm.data(), dacc.data(), n); });
    bench("L2 SoA +shared tile", [&] { nbodySoATiled<false><<<grid, block, smem>>>(dx.data(), dy.data(), dz.data(), dm.data(), dacc.data(), n); });
    bench("L3 +unroll", [&] { nbodySoATiled<true><<<grid, block, smem>>>(dx.data(), dy.data(), dz.data(), dm.data(), dacc.data(), n); });
    return 0;
}
