// pool_bench.cu
// 슬라이드: part5/chapter38 (Scan / Prefix Sum)
// 호스트 스테이징 버퍼의 "할당 + H2D 전송" 비용을 경로별로 분리 측정한다.
//   pageable        : 일반 malloc (핀 아님) — 전송 느림
//   pinned(no-pool) : 매 반복 cudaMallocHost/cudaFreeHost — 할당 비용 큼
//   pinned(pool)    : PinnedMemoryPool 재사용 — 할당 비용 제거
// 할당 비용까지 재야 하므로 CPU wall-clock(chrono)으로 측정한다.
#include "../common/raii.cuh"
#include "../common/pool.cuh"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using Clock = std::chrono::high_resolution_clock;

static double runPageable(int N, int iters, int* d_dst) {
    auto t0 = Clock::now();
    for (int it = 0; it < iters; ++it) {
        std::vector<int> h(N);                       // pageable (핀 아님)
        for (int i = 0; i < N; ++i) h[i] = i;
        CHECK_CUDA(cudaMemcpy(d_dst, h.data(), N * sizeof(int), cudaMemcpyHostToDevice));
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count() / iters;
}

static double runPinnedNoPool(int N, int iters, int* d_dst) {
    auto t0 = Clock::now();
    for (int it = 0; it < iters; ++it) {
        int* h = nullptr;
        CHECK_CUDA(cudaMallocHost(&h, N * sizeof(int)));   // 매 반복 새 핀 할당
        for (int i = 0; i < N; ++i) h[i] = i;
        CHECK_CUDA(cudaMemcpy(d_dst, h, N * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaFreeHost(h));
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count() / iters;
}

static double runPinnedPool(int N, int iters, int* d_dst) {
    auto t0 = Clock::now();
    for (int it = 0; it < iters; ++it) {
        PinnedBuffer<int> h(N);                      // 풀에서 재사용 (해제 시 풀로 반환)
        for (int i = 0; i < N; ++i) h[i] = i;
        CHECK_CUDA(cudaMemcpy(d_dst, h.data(), N * sizeof(int), cudaMemcpyHostToDevice));
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count() / iters;
}

int main(int argc, char** argv) {
    int N     = (argc > 1) ? std::atoi(argv[1]) : (1 << 22);   // 기본 4M ints
    int iters = (argc > 2) ? std::atoi(argv[2]) : 100;

    DeviceBuffer<int> d(N);
    { PinnedBuffer<int> warm(N); }                   // 풀 워밍업(한 블록 확보)

    double pageable = runPageable(N, iters, d.data());
    double noPool   = runPinnedNoPool(N, iters, d.data());
    double pool     = runPinnedPool(N, iters, d.data());

    double bytes = static_cast<double>(N) * sizeof(int);
    std::printf("=== 호스트 스테이징: 할당+H2D 비용 (N=%d, %.1f MB, %d회 평균) ===\n",
                N, bytes / 1.0e6, iters);
    std::printf("%-24s %10s %12s\n", "경로", "ms/iter", "GB/s");
    std::printf("%s\n", std::string(48, '-').c_str());
    auto row = [&](const char* name, double ms) {
        std::printf("%-24s %10.4f %12.2f\n", name, ms, bytes / (ms / 1.0e3) / 1.0e9);
    };
    row("pageable (no pin)", pageable);
    row("pinned (no pool)", noPool);
    row("pinned (pool)", pool);
    return 0;
}
