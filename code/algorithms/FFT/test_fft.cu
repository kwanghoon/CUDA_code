// test_fft.cu
// 슬라이드: part5/chapter41 (FFT)
// GoogleTest: radix-2 FFT 결과를 naive DFT 결과와 비교 (같은 변환).
#include <gtest/gtest.h>

#include "../common/raii.cuh"
#include "fft_kernels.cuh"

#include <cmath>
#include <vector>

TEST(FftTest, FftMatchesDft) {
    const int N = 512, bits = 9;
    std::vector<float2> in(N);
    for (int i = 0; i < N; ++i)
        in[i] = make_float2(std::sin(0.1f * i) + 0.5f * std::cos(0.03f * i), 0.0f);

    DeviceBuffer<float2> dIn(N), dDft(N), dFft(N);
    dIn.copyFromHost(in.data());

    DeviceBuffer<float2> dFast(N);
    dftNaive<<<1, N>>>(dIn.data(), dDft.data(), N);
    fftShared<false><<<1, N / 2, N * sizeof(float2)>>>(dIn.data(), dFft.data(), N, bits);
    fftShared<true> <<<1, N / 2, N * sizeof(float2)>>>(dIn.data(), dFast.data(), N, bits);
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float2> dft(N), fft(N), fast(N);
    dDft.copyToHost(dft.data());
    dFft.copyToHost(fft.data());
    dFast.copyToHost(fast.data());

    for (int k = 0; k < N; ++k) {
        float mag = std::sqrt(dft[k].x * dft[k].x + dft[k].y * dft[k].y);
        float dx = fft[k].x - dft[k].x, dy = fft[k].y - dft[k].y;
        EXPECT_LT(std::sqrt(dx * dx + dy * dy), 1e-2f * (1.0f + mag)) << "L1 k=" << k;
        // L2 fast twiddle: __sincosf 근사라 허용오차 살짝 크게
        float fx = fast[k].x - dft[k].x, fy = fast[k].y - dft[k].y;
        EXPECT_LT(std::sqrt(fx * fx + fy * fy), 2e-2f * (1.0f + mag)) << "L2 k=" << k;
    }
}

TEST(FftTest, SoAMatchesDft) {
    const int N = 512, bits = 9;
    std::vector<float> inRe(N), inIm(N, 0.0f);
    for (int i = 0; i < N; ++i) inRe[i] = std::sin(0.1f * i) + 0.5f * std::cos(0.03f * i);
    std::vector<float2> in(N);
    for (int i = 0; i < N; ++i) in[i] = make_float2(inRe[i], 0.0f);

    DeviceBuffer<float2> dIn(N), dDft(N);
    DeviceBuffer<float> dInRe(N), dInIm(N), dOutRe(N), dOutIm(N);
    dIn.copyFromHost(in.data()); dInRe.copyFromHost(inRe.data()); dInIm.copyFromHost(inIm.data());
    dftNaive<<<1, N>>>(dIn.data(), dDft.data(), N);
    fftSharedSoA<<<1, N / 2, 2 * N * sizeof(float)>>>(dInRe.data(), dInIm.data(), dOutRe.data(), dOutIm.data(), N, bits);
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float2> dft(N); std::vector<float> outRe(N), outIm(N);
    dDft.copyToHost(dft.data()); dOutRe.copyToHost(outRe.data()); dOutIm.copyToHost(outIm.data());
    for (int k = 0; k < N; ++k) {
        float mag = std::sqrt(dft[k].x * dft[k].x + dft[k].y * dft[k].y);
        float dx = outRe[k] - dft[k].x, dy = outIm[k] - dft[k].y;
        EXPECT_LT(std::sqrt(dx * dx + dy * dy), 1e-2f * (1.0f + mag)) << "SoA k=" << k;
    }
}
