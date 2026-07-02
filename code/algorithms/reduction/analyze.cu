// analyze.cu
// reduction을 호스트(CPU)와 비교하고 Amdahl/Gustafson/roofline로 분석.
#include "../common/raii.cuh"
#include "../common/analysis.cuh"
#include "reduction_kernels.cuh"
#include "reduction_metric.cuh"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using Clock = std::chrono::high_resolution_clock;

static double cpuSumMs(const std::vector<int>& in, int iters) {
    volatile long sink = 0;
    auto t0 = Clock::now();
    for (int it = 0; it < iters; ++it) {
        long s = 0;
        for (int x : in) s += x;
        sink += s;
    }
    (void)sink;
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count() / iters;
}

struct Split { int N; double cpu, h2d, ker, d2h; };

static int gridFor(int n) { int g = (n + RBLOCK - 1) / RBLOCK; return g > 1024 ? 1024 : g; }

static Split measure(long req, int iters) {
    int N = static_cast<int>(req);
    if (N < 1) N = 1;

    auto h_in = reduceMakeInput(N);
    int h_out = 0;
    DeviceBuffer<int> d_in(N), d_out(1);
    d_in.copyFromHost(h_in.data());
    reduceComposed<true, true, true, true><<<gridFor(N), RBLOCK>>>(d_in.data(), d_out.data(), N);
    CHECK_CUDA(cudaDeviceSynchronize());

    GpuTimer t;
    t.start(); for (int i = 0; i < iters; ++i) d_in.copyFromHost(h_in.data());
    double h2d = t.stop() / iters;
    t.start(); for (int i = 0; i < iters; ++i) reduceComposed<true, true, true, true><<<gridFor(N), RBLOCK>>>(d_in.data(), d_out.data(), N);
    double ker = t.stop() / iters;
    t.start(); for (int i = 0; i < iters; ++i) d_out.copyToHost(&h_out);
    double d2h = t.stop() / iters;

    double cpu = cpuSumMs(h_in, iters);
    return {N, cpu, h2d, ker, d2h};
}

int main(int argc, char** argv) {
    long req  = (argc > 1) ? std::atol(argv[1]) : (1L << 22);
    int  iters = (argc > 2) ? std::atoi(argv[2]) : 50;

    Split m = measure(req, iters);
    std::printf("=== Reduction: GPU vs Host — Amdahl/Gustafson/roofline (N=%d, %d회) ===\n", m.N, iters);
    printAmdahlAnalysis(m.cpu, m.h2d, m.ker, m.d2h, static_cast<double>(m.N) * sizeof(int));

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
