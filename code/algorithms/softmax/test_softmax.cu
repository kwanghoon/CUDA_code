// test_softmax.cu
// 슬라이드: part5/chapter44 (Softmax)
// GoogleTest: 각 softmax 레벨을 CPU 기준과 비교 (verifyApprox).
#include <gtest/gtest.h>

#include "../common/raii.cuh"
#include "../common/verify.cuh"
#include "softmax_kernels.cuh"

#include <cmath>
#include <vector>

static constexpr int ROWS = 64, COLS = 1024;

static std::vector<float> makeRef(std::vector<float>& in) {
    size_t sz = (size_t)ROWS * COLS;
    in.resize(sz);
    unsigned s = 3u;
    for (size_t i = 0; i < sz; ++i) { s = s * 1103515245u + 12345u; in[i] = (((s >> 16) & 0xff) / 255.0f) - 0.5f; }
    std::vector<float> ref(sz);
    for (int r = 0; r < ROWS; ++r) {
        const float* x = &in[(size_t)r * COLS]; float* y = &ref[(size_t)r * COLS];
        float m = -1e30f; for (int c = 0; c < COLS; ++c) m = std::fmax(m, x[c]);
        float l = 0; for (int c = 0; c < COLS; ++c) l += std::exp(x[c] - m);
        for (int c = 0; c < COLS; ++c) y[c] = std::exp(x[c] - m) / l;
    }
    return ref;
}

template <typename Launch>
static std::vector<float> run(const std::vector<float>& in, Launch launch) {
    DeviceBuffer<float> d_in(in.size()), d_out(in.size());
    d_in.copyFromHost(in.data());
    launch(d_in.data(), d_out.data());
    CHECK_CUDA(cudaDeviceSynchronize());
    std::vector<float> out(in.size()); d_out.copyToHost(out.data());
    return out;
}

TEST(SoftmaxTest, NoCacheMatchesCpu) {
    std::vector<float> in; auto ref = makeRef(in);
    auto o = run(in, [&](const float* i, float* o) { softmaxComposed<false, false, false><<<ROWS, SM_BLK>>>(i, o, ROWS, COLS); });
    EXPECT_TRUE(verifyApprox(o, ref, 1e-3).ok);
}
TEST(SoftmaxTest, CacheRowMatchesCpu) {
    std::vector<float> in; auto ref = makeRef(in);
    size_t sh = (size_t)COLS * sizeof(float);
    auto o = run(in, [&](const float* i, float* o) { softmaxComposed<true, false, false><<<ROWS, SM_BLK, sh>>>(i, o, ROWS, COLS); });
    EXPECT_TRUE(verifyApprox(o, ref, 1e-3).ok);
}
TEST(SoftmaxTest, WarpReduceMatchesCpu) {
    std::vector<float> in; auto ref = makeRef(in);
    size_t sh = (size_t)COLS * sizeof(float);
    auto o = run(in, [&](const float* i, float* o) { softmaxComposed<true, true, false><<<ROWS, SM_BLK, sh>>>(i, o, ROWS, COLS); });
    EXPECT_TRUE(verifyApprox(o, ref, 1e-3).ok);
}
TEST(SoftmaxTest, FastDivMatchesCpu) {
    std::vector<float> in; auto ref = makeRef(in);
    size_t sh = (size_t)COLS * sizeof(float);
    auto o = run(in, [&](const float* i, float* o) { softmaxComposed<true, true, true><<<ROWS, SM_BLK, sh>>>(i, o, ROWS, COLS); });
    EXPECT_TRUE(verifyApprox(o, ref, 2e-3).ok);   // __fdividef 근사라 허용오차 살짝 완화
}
