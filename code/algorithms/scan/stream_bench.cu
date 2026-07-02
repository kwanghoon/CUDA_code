// stream_bench.cu
// 슬라이드: part5/chapter38 (Scan / Prefix Sum)
// 단일 스트림(순차 H2D→kernel→D2H) vs 멀티 스트림(청크 오버랩) end-to-end 비교.
// pinned 호스트 메모리로 async 전송, 세그먼트를 스트림에 분배해 전송/계산을 겹친다.
#include "../common/raii.cuh"
#include "scan_kernels.cuh"
#include "scan_metric.cuh"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using Clock = std::chrono::high_resolution_clock;

int main(int argc, char** argv) {
    long req      = (argc > 1) ? std::atol(argv[1]) : (1L << 24);
    int  iters    = (argc > 2) ? std::atoi(argv[2]) : 20;
    int  nStreams = (argc > 3) ? std::atoi(argv[3]) : 4;

    int numSeg = static_cast<int>(req / SEG);
    if (numSeg < 1) numSeg = 1;
    int N = numSeg * SEG;

    int* h_in = nullptr; int* h_out = nullptr;   // pinned (async 전송용)
    CHECK_CUDA(cudaMallocHost(&h_in,  N * sizeof(int)));
    CHECK_CUDA(cudaMallocHost(&h_out, N * sizeof(int)));
    for (int i = 0; i < N; ++i) h_in[i] = (i % 7) - 3;

    DeviceBuffer<int> d_in(N), d_out(N);

    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    std::printf("=== Scan stream overlap (N=%d, %.1f MB, %d streams, %d회) ===\n",
                N, N * sizeof(int) / 1.0e6, nStreams, iters);
    std::printf("GPU : %s (SM %d.%d)\n\n", prop.name, prop.major, prop.minor);

    auto single = [&]() {
        CHECK_CUDA(cudaMemcpy(d_in.data(), h_in, N * sizeof(int), cudaMemcpyHostToDevice));
        scanComposed<true><<<numSeg, SEG>>>(d_in.data(), d_out.data());
        CHECK_CUDA(cudaMemcpy(h_out, d_out.data(), N * sizeof(int), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaDeviceSynchronize());
    };

    std::vector<cudaStream_t> st(nStreams);
    for (auto& s : st) CHECK_CUDA(cudaStreamCreate(&s));
    auto multi = [&]() {
        int segPer = (numSeg + nStreams - 1) / nStreams;
        for (int c = 0; c < nStreams; ++c) {
            int s0 = c * segPer;
            if (s0 >= numSeg) break;
            int sc  = (s0 + segPer <= numSeg) ? segPer : (numSeg - s0);
            int off = s0 * SEG, cnt = sc * SEG;
            cudaStream_t s = st[c];
            CHECK_CUDA(cudaMemcpyAsync(d_in.data() + off, h_in + off,
                                       cnt * sizeof(int), cudaMemcpyHostToDevice, s));
            scanComposed<true><<<sc, SEG, 0, s>>>(d_in.data() + off, d_out.data() + off);
            CHECK_CUDA(cudaMemcpyAsync(h_out + off, d_out.data() + off,
                                       cnt * sizeof(int), cudaMemcpyDeviceToHost, s));
        }
        CHECK_CUDA(cudaDeviceSynchronize());
    };

    single(); multi();

    auto timeit = [&](auto fn) {
        auto t0 = Clock::now();
        for (int i = 0; i < iters; ++i) fn();
        return std::chrono::duration<double, std::milli>(Clock::now() - t0).count() / iters;
    };
    double ms1 = timeit(single);
    double msN = timeit(multi);

    auto ref = scanCpuReference(std::vector<int>(h_in, h_in + N), numSeg);
    bool ok = true;
    for (int i = 0; i < N; ++i) if (h_out[i] != ref[i]) { ok = false; break; }

    double bytes = 2.0 * N * sizeof(int);
    std::printf("%-24s %10s %12s\n", "경로", "ms/iter", "GB/s(e2e)");
    std::printf("%s\n", std::string(48, '-').c_str());
    std::printf("%-24s %10.4f %12.2f\n", "single stream", ms1, bytes / (ms1 / 1.0e3) / 1.0e9);
    std::printf("%-24s %10.4f %12.2f\n", "multi stream",  msN, bytes / (msN / 1.0e3) / 1.0e9);
    std::printf("\nspeedup (single/multi): %.2fx   검증: %s\n", ms1 / msN, ok ? "OK" : "FAIL");

    for (auto& s : st) cudaStreamDestroy(s);
    cudaFreeHost(h_in);
    cudaFreeHost(h_out);
    return 0;
}
