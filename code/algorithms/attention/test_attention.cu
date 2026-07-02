// test_attention.cu
// 슬라이드: part10/chapter77 (Attention)
// GoogleTest: naive/flash attention을 CPU 기준과 비교 (verifyApprox, 작은 N).
#include <gtest/gtest.h>

#include "../common/raii.cuh"
#include "../common/verify.cuh"
#include "attention_kernels.cuh"

#include <cmath>
#include <vector>

static constexpr int N = 128;

static std::vector<float> makeRef(std::vector<float>& Q, std::vector<float>& K, std::vector<float>& V) {
    size_t sz = (size_t)N * HEAD_DIM;
    Q.resize(sz); K.resize(sz); V.resize(sz);
    unsigned s = 5u;
    auto rnd = [&]() { s = s * 1103515245u + 12345u; return (((s >> 16) & 0xff) / 255.0f) - 0.5f; };
    for (size_t i = 0; i < sz; ++i) { Q[i] = rnd(); K[i] = rnd(); V[i] = rnd(); }
    std::vector<float> ref(sz), sc(N);
    float scale = 1.0f / std::sqrt((float)HEAD_DIM);
    for (int i = 0; i < N; ++i) {
        float m = -1e30f;
        for (int j = 0; j < N; ++j) { float d = 0; for (int k = 0; k < HEAD_DIM; ++k) d += Q[i * HEAD_DIM + k] * K[j * HEAD_DIM + k]; sc[j] = d * scale; m = std::fmax(m, sc[j]); }
        float l = 0; for (int j = 0; j < N; ++j) { sc[j] = std::exp(sc[j] - m); l += sc[j]; }
        for (int t = 0; t < HEAD_DIM; ++t) { float a = 0; for (int j = 0; j < N; ++j) a += sc[j] * V[j * HEAD_DIM + t]; ref[i * HEAD_DIM + t] = a / l; }
    }
    return ref;
}

template <typename Launch>
static std::vector<float> run(const std::vector<float>& Q, const std::vector<float>& K,
                              const std::vector<float>& V, Launch launch) {
    size_t sz = Q.size();
    DeviceBuffer<float> dQ(sz), dK(sz), dV(sz), dO(sz);
    dQ.copyFromHost(Q.data()); dK.copyFromHost(K.data()); dV.copyFromHost(V.data());
    launch(dQ.data(), dK.data(), dV.data(), dO.data());
    CHECK_CUDA(cudaDeviceSynchronize());
    std::vector<float> O(sz); dO.copyToHost(O.data());
    return O;
}

TEST(AttentionTest, NaiveMatchesCpu) {
    std::vector<float> Q, K, V; auto ref = makeRef(Q, K, V);
    size_t sh = (size_t)(N + HEAD_DIM + HEAD_DIM) * sizeof(float);
    auto O = run(Q, K, V, [&](const float* q, const float* k, const float* v, float* o) {
        attnNaive<<<N, HEAD_DIM, sh>>>(q, k, v, o, N);
    });
    EXPECT_TRUE(verifyApprox(O, ref, 2e-2).ok);
}

TEST(AttentionTest, FlashMatchesCpu) {
    std::vector<float> Q, K, V; auto ref = makeRef(Q, K, V);
    auto O = run(Q, K, V, [&](const float* q, const float* k, const float* v, float* o) {
        attnFlash<false><<<N, HEAD_DIM>>>(q, k, v, o, N);
    });
    EXPECT_TRUE(verifyApprox(O, ref, 2e-2).ok);
}
