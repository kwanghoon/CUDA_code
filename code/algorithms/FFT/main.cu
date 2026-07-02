// main.cu
// 슬라이드: part5/chapter41 (FFT) — 배치 1D FFT. GFLOP/s(=5 N log N 기준)로 DFT vs FFT 비교.
// complex(float2)는 verify.cuh 캐스팅이 안 되므로 자체 비교(첫 신호 vs CPU DFT).
#include "../common/raii.cuh"
#include "../common/cli.cuh"
#include "fft_kernels.cuh"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

static bool verifyFFT(const std::vector<float2>& out, const std::vector<float2>& ref, float tol) {
    for (size_t k = 0; k < ref.size(); ++k) {
        float dx = out[k].x - ref[k].x, dy = out[k].y - ref[k].y;
        float mag = std::sqrt(ref[k].x * ref[k].x + ref[k].y * ref[k].y);
        if (std::sqrt(dx * dx + dy * dy) > tol * (1.0f + mag)) return false;
    }
    return true;
}

int main(int argc, char** argv) {
    BenchOptions opt = CliParser::parse(argc, argv);
    if (!opt.valid)   { CliParser::printUsage(argv[0]); return 1; }
    if (opt.showHelp) { CliParser::printUsage(argv[0]); return 0; }

    const int N = 512, bits = 9;                   // FFT 크기(2^9)
    long total = (opt.n > 0) ? opt.n : (1L << 20);
    int B = static_cast<int>(total / N); if (B < 1) B = 1;   // 배치 개수
    int iters = opt.iters;

    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    std::printf("=== FFT 벤치마크 (N=%d, batch=%d, %d회) ===\n", N, B, iters);
    std::printf("GPU : %s (SM %d.%d)\n\n", prop.name, prop.major, prop.minor);

    std::vector<float2> hin((size_t)B * N), hout((size_t)B * N);
    for (size_t i = 0; i < hin.size(); ++i)
        hin[i] = make_float2(std::sin(0.1f * (i % N)) + 0.5f * std::cos(0.03f * (i % N)), 0.0f);

    // CPU DFT 기준 (첫 신호)
    std::vector<float2> ref(N);
    for (int k = 0; k < N; ++k) {
        float re = 0.0f, im = 0.0f;
        for (int n = 0; n < N; ++n) {
            float ang = -2.0f * FFT_PI * k * n / N;
            float c = std::cos(ang), s = std::sin(ang);
            re += hin[n].x * c - hin[n].y * s;
            im += hin[n].x * s + hin[n].y * c;
        }
        ref[k] = make_float2(re, im);
    }

    DeviceBuffer<float2> dIn((size_t)B * N), dOut((size_t)B * N);
    dIn.copyFromHost(hin.data());

    double flops = (double)B * 5.0 * N * bits;     // ~5 N log2 N per transform
    float  tol   = 1.0e-2f;

    std::printf("%-22s %10s %12s %8s %10s\n", "Stage", "Time(ms)", "GFLOP/s", "Check", "Speedup");
    std::printf("%s\n", std::string(66, '-').c_str());

    double baseMs = -1.0;
    auto bench = [&](const char* name, auto launch) {
        launch();
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        dOut.copyToHost(hout.data());
        std::vector<float2> first(hout.begin(), hout.begin() + N);
        bool ok = verifyFFT(first, ref, tol);

        GpuTimer t;
        t.start();
        for (int i = 0; i < iters; ++i) launch();
        double ms = t.stop() / iters;
        double gf = flops / (ms / 1.0e3) / 1.0e9;
        if (baseMs < 0) baseMs = ms;
        std::printf("%-22s %10.4f %12.2f %8s %9.2fx\n", name, ms, gf, ok ? "OK" : "FAIL", baseMs / ms);
    };

    bench("L0 naive DFT", [&] { dftNaive<<<B, N>>>(dIn.data(), dOut.data(), N); });
    bench("L1 radix-2 FFT", [&] {
        fftShared<false><<<B, N / 2, N * sizeof(float2)>>>(dIn.data(), dOut.data(), N, bits);
    });
    bench("L2 +fast twiddle", [&] {
        fftShared<true><<<B, N / 2, N * sizeof(float2)>>>(dIn.data(), dOut.data(), N, bits);
    });

    // AoS vs SoA 케이스: 위는 복소수 float2(AoS). 아래는 real/imag 분리(SoA).
    std::vector<float> hInRe((size_t)B * N), hInIm((size_t)B * N), hOutRe((size_t)B * N), hOutIm((size_t)B * N);
    for (size_t i = 0; i < hin.size(); ++i) { hInRe[i] = hin[i].x; hInIm[i] = hin[i].y; }
    DeviceBuffer<float> dInRe((size_t)B * N), dInIm((size_t)B * N), dOutRe((size_t)B * N), dOutIm((size_t)B * N);
    dInRe.copyFromHost(hInRe.data()); dInIm.copyFromHost(hInIm.data());
    {
        auto launch = [&] {
            fftSharedSoA<<<B, N / 2, 2 * N * sizeof(float)>>>(dInRe.data(), dInIm.data(),
                                                             dOutRe.data(), dOutIm.data(), N, bits);
        };
        launch(); CHECK_CUDA(cudaDeviceSynchronize());
        dOutRe.copyToHost(hOutRe.data()); dOutIm.copyToHost(hOutIm.data());
        std::vector<float2> first(N);
        for (int k = 0; k < N; ++k) first[k] = make_float2(hOutRe[k], hOutIm[k]);
        bool ok = verifyFFT(first, ref, tol);
        GpuTimer t; t.start();
        for (int i = 0; i < iters; ++i) launch();
        double ms = t.stop() / iters;
        std::printf("%-22s %10.4f %12.2f %8s %9.2fx\n", "SoA (real/imag split)",
                    ms, flops / (ms / 1.0e3) / 1.0e9, ok ? "OK" : "FAIL", baseMs / ms);
    }
    return 0;
}
