// test_matmul.cu
// GoogleTest: 각 matmul 레벨을 CPU 기준과 비교 (verifyApprox, 작은 크기).
#include <gtest/gtest.h>

#include "../common/raii.cuh"
#include "../common/verify.cuh"
#include "matmul_kernels.cuh"

#include <vector>

static constexpr int SZ = 128;   // 16의 배수

static void cpuMatmul(const std::vector<float>& A, const std::vector<float>& B,
                      std::vector<float>& C, int n) {
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            float s = 0.0f;
            for (int k = 0; k < n; ++k) s += A[i * n + k] * B[k * n + j];
            C[i * n + j] = s;
        }
}

static std::vector<float> inputs(std::vector<float>& A, std::vector<float>& B) {
    size_t sz = (size_t)SZ * SZ;
    A.resize(sz); B.resize(sz);
    for (size_t i = 0; i < sz; ++i) { A[i] = ((i % 7) - 3) * 0.1f; B[i] = ((i % 5) - 2) * 0.1f; }
    std::vector<float> ref(sz);
    cpuMatmul(A, B, ref, SZ);
    return ref;
}

TEST(MatmulTest, NaiveMatchesCpu) {
    std::vector<float> A, B; auto ref = inputs(A, B);
    DeviceBuffer<float> dA(A.size()), dB(B.size()), dC(ref.size());
    dA.copyFromHost(A.data()); dB.copyFromHost(B.data());
    dim3 b(16, 16), g(SZ / 16, SZ / 16);
    matmulComposed<false><<<g, b>>>(dA.data(), dB.data(), dC.data(), SZ, SZ, SZ);
    CHECK_CUDA(cudaDeviceSynchronize());
    std::vector<float> out(ref.size()); dC.copyToHost(out.data());
    EXPECT_TRUE(verifyApprox(out, ref, 1e-3).ok);
}

TEST(MatmulTest, TiledMatchesCpu) {
    std::vector<float> A, B; auto ref = inputs(A, B);
    DeviceBuffer<float> dA(A.size()), dB(B.size()), dC(ref.size());
    dA.copyFromHost(A.data()); dB.copyFromHost(B.data());
    dim3 b(16, 16), g(SZ / 16, SZ / 16);
    matmulComposed<true><<<g, b>>>(dA.data(), dB.data(), dC.data(), SZ, SZ, SZ);
    CHECK_CUDA(cudaDeviceSynchronize());
    std::vector<float> out(ref.size()); dC.copyToHost(out.data());
    EXPECT_TRUE(verifyApprox(out, ref, 1e-3).ok);
}

TEST(MatmulTest, RegTiledMatchesCpu) {
    std::vector<float> A, B; auto ref = inputs(A, B);
    DeviceBuffer<float> dA(A.size()), dB(B.size()), dC(ref.size());
    dA.copyFromHost(A.data()); dB.copyFromHost(B.data());
    dim3 g((SZ + 63) / 64, (SZ + 63) / 64);
    sgemmRegTiled<64, 64, 8, 4, 4><<<g, 256>>>(dA.data(), dB.data(), dC.data(), SZ, SZ, SZ);
    CHECK_CUDA(cudaDeviceSynchronize());
    std::vector<float> out(ref.size()); dC.copyToHost(out.data());
    EXPECT_TRUE(verifyApprox(out, ref, 1e-3).ok);
}

TEST(MatmulTest, RegTiledAsyncMatchesCpu) {
    std::vector<float> A, B; auto ref = inputs(A, B);
    DeviceBuffer<float> dA(A.size()), dB(B.size()), dC(ref.size());
    dA.copyFromHost(A.data()); dB.copyFromHost(B.data());
    dim3 g((SZ + 63) / 64, (SZ + 63) / 64);
    sgemmRegTiledAsync<64, 64, 8, 4, 4><<<g, 256>>>(dA.data(), dB.data(), dC.data(), SZ, SZ, SZ);
    CHECK_CUDA(cudaDeviceSynchronize());
    std::vector<float> out(ref.size()); dC.copyToHost(out.data());
    EXPECT_TRUE(verifyApprox(out, ref, 1e-3).ok);
}

TEST(MatmulTest, WmmaMatchesCpu) {
    std::vector<float> A, B; auto ref = inputs(A, B);
    std::vector<half> Ah(A.size()), Bh(B.size());
    for (size_t i = 0; i < A.size(); ++i) { Ah[i] = __float2half(A[i]); Bh[i] = __float2half(B[i]); }
    DeviceBuffer<half> dA(A.size()), dB(B.size());
    DeviceBuffer<float> dC(ref.size());
    dA.copyFromHost(Ah.data()); dB.copyFromHost(Bh.data());
    dim3 bw(128, 4), gw((SZ + 63) / 64, (SZ + 63) / 64);
    kWmma<<<gw, bw>>>(dA.data(), dB.data(), dC.data(), SZ, SZ, SZ);
    CHECK_CUDA(cudaDeviceSynchronize());
    std::vector<float> out(ref.size()); dC.copyToHost(out.data());
    EXPECT_TRUE(verifyApprox(out, ref, 3e-2).ok);   // FP16 허용오차
}
