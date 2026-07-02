// unified_bench.cu
// 슬라이드: part5/chapter38 (Scan) — 호스트/디바이스 메모리 경로 비교 (end-to-end wall-clock).
// 같은 커널(scale)을 "입력 준비 → GPU 가용화 → 커널 → 결과 회수"까지 경로별로 잰다.
//   pageable   : malloc + cudaMemcpy (핀 아님) — 전송 느림
//   pinned     : cudaMallocHost — DMA 직전송, 전송 빠름
//   managed    : cudaMallocManaged + cudaMemAdvise + cudaMemPrefetchAsync (지원 시)
//   zero-copy  : cudaHostAlloc(Mapped) + cudaHostGetDevicePointer — 커널이 호스트메모리 직접 접근
// Jetson(통합 GPU)은 물리 메모리를 CPU/GPU가 공유 → managed/zero-copy가 복사 없이 유리할 수 있다.
#include "../common/raii.cuh"

#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

using Clock = std::chrono::high_resolution_clock;

__global__ void scaleK(float* __restrict__ a, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        a[i] = a[i] * 2.0f + 1.0f;
}

static double ms(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

int main(int argc, char** argv) {
    int N = (argc > 1) ? std::atoi(argv[1]) : (1 << 22);
    int iters = (argc > 2) ? std::atoi(argv[2]) : 100;
    int blk = 256, grid = (N + blk - 1) / blk;
    if (grid > 1024) grid = 1024;

    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    int concurrentManaged = 0, canMapHost = 0, integrated = prop.integrated;
    cudaDeviceGetAttribute(&concurrentManaged, cudaDevAttrConcurrentManagedAccess, 0);
    cudaDeviceGetAttribute(&canMapHost, cudaDevAttrCanMapHostMemory, 0);
    std::printf("=== Unified/Host 메모리 경로 벤치 (N=%d, %d회) ===\n", N, iters);
    std::printf("GPU : %s (SM %d.%d, integrated=%d, concurrentManaged=%d, canMapHost=%d)\n\n",
                prop.name, prop.major, prop.minor, integrated, concurrentManaged, canMapHost);
    std::printf("%-30s %12s %10s  %s\n", "경로", "ms/iter", "상대", "비고");
    std::printf("%s\n", std::string(72, '-').c_str());

    double base = -1.0;
    auto report = [&](const char* name, double perIter, const char* note) {
        if (base < 0) base = perIter;
        std::printf("%-30s %12.4f %9.2fx  %s\n", name, perIter, base / perIter, note);
    };

    // 1) pageable: malloc + cudaMemcpy 왕복
    {
        std::vector<float> h(N);
        float* d = nullptr; CHECK_CUDA(cudaMalloc(&d, N * sizeof(float)));
        auto t0 = Clock::now();
        for (int it = 0; it < iters; ++it) {
            for (int i = 0; i < N; ++i) h[i] = 1.0f;
            CHECK_CUDA(cudaMemcpy(d, h.data(), N * sizeof(float), cudaMemcpyHostToDevice));
            scaleK<<<grid, blk>>>(d, N);
            CHECK_CUDA(cudaMemcpy(h.data(), d, N * sizeof(float), cudaMemcpyDeviceToHost));
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        report("pageable (malloc+memcpy)", ms(t0) / iters, "핀 아님 → 전송 느림");
        cudaFree(d);
    }

    // 2) pinned: cudaMallocHost
    {
        float* h = nullptr; CHECK_CUDA(cudaMallocHost(&h, N * sizeof(float)));
        float* d = nullptr; CHECK_CUDA(cudaMalloc(&d, N * sizeof(float)));
        auto t0 = Clock::now();
        for (int it = 0; it < iters; ++it) {
            for (int i = 0; i < N; ++i) h[i] = 1.0f;
            CHECK_CUDA(cudaMemcpy(d, h, N * sizeof(float), cudaMemcpyHostToDevice));
            scaleK<<<grid, blk>>>(d, N);
            CHECK_CUDA(cudaMemcpy(h, d, N * sizeof(float), cudaMemcpyDeviceToHost));
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        report("pinned (cudaMallocHost)", ms(t0) / iters, "DMA 직전송 → 전송 빠름");
        cudaFree(d); cudaFreeHost(h);
    }

    // 3) managed: cudaMallocManaged + advise + prefetch (지원 시)
    {
        float* m = nullptr; CHECK_CUDA(cudaMallocManaged(&m, N * sizeof(float)));
        cudaMemAdvise(m, N * sizeof(float), cudaMemAdviseSetPreferredLocation, 0);   // GPU 선호
        auto t0 = Clock::now();
        for (int it = 0; it < iters; ++it) {
            for (int i = 0; i < N; ++i) m[i] = 1.0f;                 // 호스트에서 초기화
            if (concurrentManaged) cudaMemPrefetchAsync(m, N * sizeof(float), 0);    // → GPU
            scaleK<<<grid, blk>>>(m, N);
            if (concurrentManaged) cudaMemPrefetchAsync(m, N * sizeof(float), cudaCpuDeviceId);
            CHECK_CUDA(cudaDeviceSynchronize());                     // 결과 호스트 가용
        }
        report("managed (+advise+prefetch)", ms(t0) / iters,
               concurrentManaged ? "복사 대신 마이그레이션" : "prefetch 미지원 → advise만");
        cudaFree(m);
    }

    // 4) zero-copy: 매핑된 호스트메모리를 커널이 직접 접근 (통합 GPU서 특히 유리)
    if (canMapHost) {
        float* h = nullptr;
        CHECK_CUDA(cudaHostAlloc(&h, N * sizeof(float), cudaHostAllocMapped));
        float* d = nullptr; CHECK_CUDA(cudaHostGetDevicePointer(&d, h, 0));
        auto t0 = Clock::now();
        for (int it = 0; it < iters; ++it) {
            for (int i = 0; i < N; ++i) h[i] = 1.0f;
            scaleK<<<grid, blk>>>(d, N);            // 명시적 복사 없음
            CHECK_CUDA(cudaDeviceSynchronize());    // 커널이 h를 직접 기록
        }
        report("zero-copy (Mapped host)", ms(t0) / iters,
               integrated ? "통합 GPU → 물리 공유, 복사 0" : "PCIe 매핑(재사용 적을 때만 이득)");
        cudaFreeHost(h);
    } else {
        std::printf("%-30s %12s %10s  %s\n", "zero-copy (Mapped host)", "-", "-", "canMapHost=0 → 건너뜀");
    }

    std::printf("\n요약: 전송이 지배적이면 pinned>pageable, 재사용/통합GPU면 managed·zero-copy가 복사를 없앤다.\n");
    return 0;
}
