// main.cu
// 슬라이드: part5/chapter44 (ML Layers) — row-wise softmax 벤치. 자체 루프, GB/s + verifyApprox.
#include "../common/raii.cuh"
#include "../common/verify.cuh"
#include "../common/cli.cuh"
#include "softmax_kernels.cuh"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

static void cpuSoftmax(const std::vector<float>& in, std::vector<float>& out, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        const float* x = &in[(size_t)r * cols];
        float* y = &out[(size_t)r * cols];
        float m = -1e30f; for (int c = 0; c < cols; ++c) m = std::fmax(m, x[c]);
        float l = 0.0f;   for (int c = 0; c < cols; ++c) l += std::exp(x[c] - m);
        for (int c = 0; c < cols; ++c) y[c] = std::exp(x[c] - m) / l;
    }
}

int main(int argc, char** argv) {
    BenchOptions opt = CliParser::parse(argc, argv);
    if (!opt.valid)   { CliParser::printUsage(argv[0]); return 1; }
    if (opt.showHelp) { CliParser::printUsage(argv[0]); return 0; }

    const int cols = 1024;
    int rows = (opt.n > 0 && opt.n <= (1 << 20)) ? static_cast<int>(opt.n) : 8192;
    int iters = opt.iters;

    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    std::printf("=== Softmax 벤치마크 (rows=%d, cols=%d, %d회) ===\n", rows, cols, iters);
    std::printf("GPU : %s (SM %d.%d)\n\n", prop.name, prop.major, prop.minor);

    size_t sz = (size_t)rows * cols;
    std::vector<float> hin(sz), hout(sz), ref(sz);
    unsigned s = 77u;
    for (size_t i = 0; i < sz; ++i) { s = s * 1103515245u + 12345u; hin[i] = (((s >> 16) & 0xff) / 255.0f) - 0.5f; }
    cpuSoftmax(hin, ref, rows, cols);

    DeviceBuffer<float> d_in(sz), d_out(sz);
    d_in.copyFromHost(hin.data());

    double bytes = 2.0 * sz * sizeof(float);
    float  tol   = 1.0e-3f;
    size_t sh    = (size_t)cols * sizeof(float);

    std::printf("%-28s %10s %12s %8s %10s\n", "Stage", "Time(ms)", "GB/s", "Check", "Speedup");
    std::printf("%s\n", std::string(72, '-').c_str());

    double baseMs = -1.0;
    auto bench = [&](const char* name, size_t shBytes, auto launch) {
        launch();
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        d_out.copyToHost(hout.data());
        auto vr = verifyApprox(hout, ref, tol);
        GpuTimer t;
        t.start();
        for (int i = 0; i < iters; ++i) launch();
        double ms = t.stop() / iters;
        double gb = bytes / (ms / 1.0e3) / 1.0e9;
        (void)shBytes;
        if (baseMs < 0) baseMs = ms;
        std::printf("%-28s %10.4f %12.2f %8s %9.2fx\n", name, ms, gb, vr.ok ? "OK" : "FAIL", baseMs / ms);
    };

    bench("L0 serial (thread/row)", 0, [&] {
        softmaxSerial<<<(rows + SM_BLK - 1) / SM_BLK, SM_BLK>>>(d_in.data(), d_out.data(), rows, cols);
    });
    bench("L1 block/row (parallel)", 0, [&] {
        softmaxComposed<false, false, false><<<rows, SM_BLK>>>(d_in.data(), d_out.data(), rows, cols);
    });
    bench("L2 +cache row +warp", sh, [&] {
        softmaxComposed<true, true, false><<<rows, SM_BLK, sh>>>(d_in.data(), d_out.data(), rows, cols);
    });
    bench("L3 +fast reciprocal", sh, [&] {   // 고도화: 행별 나눗셈 → __fdividef 역수 곱
        softmaxComposed<true, true, true><<<rows, SM_BLK, sh>>>(d_in.data(), d_out.data(), rows, cols);
    });
    return 0;
}
