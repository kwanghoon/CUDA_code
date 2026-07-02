// test_spmv.cu
// 슬라이드: part5/chapter40 (SpMV)
// GoogleTest: CSR SpMV 레벨을 CPU 기준과 비교 (verifyApprox).
#include <gtest/gtest.h>

#include "../common/raii.cuh"
#include "../common/verify.cuh"
#include "spmv_kernels.cuh"

#include <algorithm>
#include <vector>

struct Csr {
    int M, N, K;
    std::vector<int> rowPtr, colIdx;
    std::vector<float> vals, x, ref;
};

static Csr makeCsr(int M) {
    Csr c; c.M = M; c.N = M; c.K = 16;
    long nnz = (long)M * c.K;
    c.rowPtr.resize(M + 1); c.colIdx.resize(nnz); c.vals.resize(nnz); c.x.resize(M); c.ref.resize(M);
    for (int r = 0; r <= M; ++r) c.rowPtr[r] = r * c.K;
    unsigned s = 7u;
    for (long i = 0; i < nnz; ++i) { s = s * 1103515245u + 12345u; c.colIdx[i] = (int)((s >> 9) % M); c.vals[i] = (((s >> 16) & 0xff) / 255.0f) - 0.5f; }
    for (int i = 0; i < M; ++i) { s = s * 1103515245u + 12345u; c.x[i] = (((s >> 16) & 0xff) / 255.0f) - 0.5f; }
    for (int r = 0; r < M; ++r) { float a = 0; for (int k = c.rowPtr[r]; k < c.rowPtr[r + 1]; ++k) a += c.vals[k] * c.x[c.colIdx[k]]; c.ref[r] = a; }
    return c;
}

template <typename Launch>
static std::vector<float> run(const Csr& c, Launch launch) {
    DeviceBuffer<int> dRow(c.M + 1), dCol(c.colIdx.size());
    DeviceBuffer<float> dVal(c.vals.size()), dX(c.N), dY(c.M);
    dRow.copyFromHost(c.rowPtr.data()); dCol.copyFromHost(c.colIdx.data());
    dVal.copyFromHost(c.vals.data());   dX.copyFromHost(c.x.data());
    launch(dRow.data(), dCol.data(), dVal.data(), dX.data(), dY.data(), c.M);
    CHECK_CUDA(cudaDeviceSynchronize());
    std::vector<float> y(c.M); dY.copyToHost(y.data());
    return y;
}

TEST(SpmvTest, ScalarMatchesCpu) {
    auto c = makeCsr(4096);
    auto y = run(c, [](const int* rp, const int* ci, const float* v, const float* x, float* y, int M) {
        spmvScalar<<<(M + 255) / 256, 256>>>(rp, ci, v, x, y, M);
    });
    EXPECT_TRUE(verifyApprox(y, c.ref, 1e-2).ok);
}
TEST(SpmvTest, VectorMatchesCpu) {
    auto c = makeCsr(4096);
    auto y = run(c, [](const int* rp, const int* ci, const float* v, const float* x, float* y, int M) {
        spmvVector<false><<<(M * 32 + 127) / 128, 128>>>(rp, ci, v, x, y, M);
    });
    EXPECT_TRUE(verifyApprox(y, c.ref, 1e-2).ok);
}
TEST(SpmvTest, VectorLdgMatchesCpu) {
    auto c = makeCsr(4096);
    auto y = run(c, [](const int* rp, const int* ci, const float* v, const float* x, float* y, int M) {
        spmvVector<true><<<(M * 32 + 127) / 128, 128>>>(rp, ci, v, x, y, M);
    });
    EXPECT_TRUE(verifyApprox(y, c.ref, 1e-2).ok);
}
TEST(SpmvTest, EllMatchesCpu) {
    auto c = makeCsr(4096);
    int maxNnz = 0;
    for (int r = 0; r < c.M; ++r) maxNnz = std::max(maxNnz, c.rowPtr[r + 1] - c.rowPtr[r]);
    std::vector<int>   ellCol((long)maxNnz * c.M, -1);
    std::vector<float> ellVal((long)maxNnz * c.M, 0.0f);
    for (int r = 0; r < c.M; ++r) {
        int j = 0;
        for (int k = c.rowPtr[r]; k < c.rowPtr[r + 1]; ++k, ++j) {
            ellCol[(long)j * c.M + r] = c.colIdx[k];
            ellVal[(long)j * c.M + r] = c.vals[k];
        }
    }
    DeviceBuffer<int>   dCol((long)maxNnz * c.M);
    DeviceBuffer<float> dVal((long)maxNnz * c.M), dX(c.N), dY(c.M);
    dCol.copyFromHost(ellCol.data()); dVal.copyFromHost(ellVal.data()); dX.copyFromHost(c.x.data());
    spmvEll<<<(c.M + 255) / 256, 256>>>(dCol.data(), dVal.data(), dX.data(), dY.data(), c.M, maxNnz);
    CHECK_CUDA(cudaDeviceSynchronize());
    std::vector<float> y(c.M); dY.copyToHost(y.data());
    EXPECT_TRUE(verifyApprox(y, c.ref, 1e-2).ok);
}
TEST(SpmvTest, AoSMatchesCpu) {
    auto c = makeCsr(4096);
    std::vector<SpElem> aos(c.colIdx.size());
    for (size_t k = 0; k < aos.size(); ++k) { aos[k].val = c.vals[k]; aos[k].col = c.colIdx[k]; }
    DeviceBuffer<int> dRow(c.M + 1); DeviceBuffer<SpElem> dAoS(aos.size());
    DeviceBuffer<float> dX(c.N), dY(c.M);
    dRow.copyFromHost(c.rowPtr.data()); dAoS.copyFromHost(aos.data()); dX.copyFromHost(c.x.data());
    spmvVectorAoS<<<(c.M * 32 + 127) / 128, 128>>>(dRow.data(), dAoS.data(), dX.data(), dY.data(), c.M);
    CHECK_CUDA(cudaDeviceSynchronize());
    std::vector<float> y(c.M); dY.copyToHost(y.data());
    EXPECT_TRUE(verifyApprox(y, c.ref, 1e-2).ok);
}
