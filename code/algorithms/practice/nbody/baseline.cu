// 실습: N-body (중력) 최적화 — BASELINE (여기서 시작)
// 목표: O(N²) N-body 를 ncu로 병목을 찾아 AoS→SoA 레이아웃 + shared 타일링으로 최적화.
// 핵심 케이스: 데이터 레이아웃 AoS(Array of Structures) vs SoA(Structure of Arrays).
//   빌드:  nvcc -O3 -arch=sm_87 baseline.cu -o baseline
//   프로파일: ncu --set full ./baseline
//   STEP 1  l1tex__t_sector_hit_rate.pct / gld 효율 → AoS strided → SoA coalesced (x[],y[],z[],m[])
//   STEP 2  dram__bytes.sum 큼 → SHARED TILING (body 타일 재사용, 전역 O(N²)→O(N²/blockDim))
//   STEP 3  sm__throughput / sm__pipe_fp32_cycles_active → #pragma unroll, fast-math(rsqrtf)
//   타일링 후엔 compute-bound (rsqrtf 연산 밀도↑).
#include "../../common/raii.cuh"
#include <cstdio>
#include <vector>

// BASELINE 레이아웃: AoS (Array of Structures) — 한 body의 필드가 뭉쳐 있음
struct Body { float x, y, z, m; };

// naive: 스레드 i가 모든 j를 전역에서 반복 로드 (타일링 없음), AoS 필드 접근(strided)
__global__ void nbodyAoS(const Body* __restrict__ b, float3* __restrict__ acc, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float xi = b[i].x, yi = b[i].y, zi = b[i].z;      // AoS: 자기 body 필드(인접 스레드와 16B 간격)
    float ax = 0, ay = 0, az = 0;
    for (int j = 0; j < n; ++j) {                     // O(N²), 전역 반복 로드
        float dx = b[j].x - xi, dy = b[j].y - yi, dz = b[j].z - zi;
        float inv = rsqrtf(dx * dx + dy * dy + dz * dz + 1e-4f);
        float s = b[j].m * inv * inv * inv;
        ax += dx * s; ay += dy * s; az += dz * s;
    }
    acc[i] = make_float3(ax, ay, az);
}

int main(int argc, char** argv) {
    int n = (argc > 1) ? atoi(argv[1]) : (1 << 14);   // 16384 bodies
    std::vector<Body> h(n);
    unsigned s = 7u;
    for (int i = 0; i < n; ++i) {
        s = s * 1103515245u + 12345u; h[i].x = ((s >> 16) & 1023) * 0.1f;
        s = s * 1103515245u + 12345u; h[i].y = ((s >> 16) & 1023) * 0.1f;
        s = s * 1103515245u + 12345u; h[i].z = ((s >> 16) & 1023) * 0.1f;
        h[i].m = 1.0f;
    }
    DeviceBuffer<Body> db(n); DeviceBuffer<float3> dacc(n);
    db.copyFromHost(h.data());

    int block = 256, grid = (n + block - 1) / block;
    nbodyAoS<<<grid, block>>>(db.data(), dacc.data(), n);
    CHECK_CUDA(cudaDeviceSynchronize());

    GpuTimer t; t.start();
    const int iters = 10;
    for (int k = 0; k < iters; ++k) nbodyAoS<<<grid, block>>>(db.data(), dacc.data(), n);
    double ms = t.stop() / iters;

    std::vector<float3> acc(n); dacc.copyToHost(acc.data());
    double gflops = 20.0 * (double)n * n / (ms / 1e3) / 1e9;   // ~20 flop/interaction
    std::printf("baseline(AoS): %.4f ms   %.1f GFLOP/s   acc[0]=(%.3f,%.3f,%.3f)\n",
                ms, gflops, acc[0].x, acc[0].y, acc[0].z);
    std::printf("→ ncu --set full ./baseline 로 STEP1(AoS→SoA)·STEP2(타일링) 을 보고 고쳐라. 정답: solution.cu\n");
    return 0;
}
