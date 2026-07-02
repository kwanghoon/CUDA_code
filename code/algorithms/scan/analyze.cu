// analyze.cu
// 슬라이드: part5/chapter38 (Scan / Prefix Sum)
// scan을 호스트(CPU)와 비교하고 Amdahl/Gustafson/roofline로 분석. 코어 analysis 재사용.
#include "../common/raii.cuh"
#include "../common/analysis.cuh"
#include "scan_kernels.cuh"
#include "scan_metric.cuh"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using Clock = std::chrono::high_resolution_clock;

static double cpuScanMs(const std::vector<int>& in, int numSeg, int iters) {
    std::vector<int> out(in.size());
    volatile long sink = 0;
    auto t0 = Clock::now();
    for (int it = 0; it < iters; ++it) {
        for (int s = 0; s < numSeg; ++s) {
            int acc = 0, base = s * SEG;
            for (int i = 0; i < SEG; ++i) { acc += in[base + i]; out[base + i] = acc; }
        }
        sink += out[0];
    }
    (void)sink;
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count() / iters;
}

struct Split { int N; double cpu, h2d, ker, d2h; };

static Split measure(long req, int iters) {
    int numSeg = static_cast<int>(req / SEG);
    if (numSeg < 1) numSeg = 1;
    int N = numSeg * SEG;

    auto h_in = scanMakeInput(N);
    std::vector<int> h_out(N);
    DeviceBuffer<int> d_in(N), d_out(N);
    d_in.copyFromHost(h_in.data());
    scanComposed<true><<<numSeg, SEG>>>(d_in.data(), d_out.data());
    CHECK_CUDA(cudaDeviceSynchronize());

    GpuTimer t;
    t.start(); for (int i = 0; i < iters; ++i) d_in.copyFromHost(h_in.data());
    double h2d = t.stop() / iters;
    t.start(); for (int i = 0; i < iters; ++i) scanComposed<true><<<numSeg, SEG>>>(d_in.data(), d_out.data());
    double ker = t.stop() / iters;
    t.start(); for (int i = 0; i < iters; ++i) d_out.copyToHost(h_out.data());
    double d2h = t.stop() / iters;

    double cpu = cpuScanMs(h_in, numSeg, iters);
    return {N, cpu, h2d, ker, d2h};
}

int main(int argc, char** argv) {
    long req  = (argc > 1) ? std::atol(argv[1]) : (1L << 22);
    int  iters = (argc > 2) ? std::atoi(argv[2]) : 50;

    Split m = measure(req, iters);
    std::printf("=== Scan: GPU vs Host — Amdahl/Gustafson/roofline (N=%d, %d회) ===\n", m.N, iters);
    printAmdahlAnalysis(m.cpu, m.h2d, m.ker, m.d2h, 2.0 * m.N * sizeof(int));

    std::printf("\nGustafson 스케일링 (N 증가 시 end-to-end speedup):\n");
    std::printf("%12s %14s %14s\n", "N", "GPU e2e(ms)", "speedup");
    std::printf("%s\n", std::string(42, '-').c_str());
    for (long n = (1L << 20); n <= (1L << 24); n <<= 1) {
        Split s = measure(n, iters);
        double e2e = s.h2d + s.ker + s.d2h;
        std::printf("%12d %14.4f %13.2fx\n", s.N, e2e, s.cpu / e2e);
    }
    return 0;
}
