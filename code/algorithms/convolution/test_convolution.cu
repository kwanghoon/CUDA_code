// test_convolution.cu
// 슬라이드: part5/chapter35 (Convolution)
// GoogleTest: 각 conv 레벨을 CPU 기준과 비교 (verifyApprox, 작은 이미지).
#include <gtest/gtest.h>

#include "../common/raii.cuh"
#include "../common/verify.cuh"
#include "convolution_kernels.cuh"

#include <algorithm>
#include <vector>

static constexpr int SZ = 128;

static std::vector<float> makeRef(std::vector<float>& in, std::vector<float>& f) {
    size_t sz = (size_t)SZ * SZ;
    in.resize(sz); f.assign(FSIZE, 1.0f / FSIZE);
    for (size_t i = 0; i < sz; ++i) in[i] = ((i % 13) - 6) * 0.1f;
    std::vector<float> ref(sz);
    for (int y = 0; y < SZ; ++y)
        for (int x = 0; x < SZ; ++x) {
            float s = 0.0f;
            for (int fy = -RADIUS; fy <= RADIUS; ++fy)
                for (int fx = -RADIUS; fx <= RADIUS; ++fx) {
                    int ix = std::min(std::max(x + fx, 0), SZ - 1);
                    int iy = std::min(std::max(y + fy, 0), SZ - 1);
                    s += in[iy * SZ + ix] * f[(fy + RADIUS) * FDIM + (fx + RADIUS)];
                }
            ref[y * SZ + x] = s;
        }
    return ref;
}

static std::vector<float> run(bool useConst, bool useShared,
                              const std::vector<float>& in, const std::vector<float>& f) {
    size_t sz = in.size();
    DeviceBuffer<float> din(sz), dout(sz), dfilter(FSIZE);
    din.copyFromHost(in.data());
    dfilter.copyFromHost(f.data());
    CHECK_CUDA(cudaMemcpyToSymbol(cFilter, f.data(), FSIZE * sizeof(float)));
    dim3 b(CTILE, CTILE), g((SZ + CTILE - 1) / CTILE, (SZ + CTILE - 1) / CTILE);
    if (!useConst && !useShared) convComposed<false, false><<<g, b>>>(din.data(), dfilter.data(), dout.data(), SZ, SZ);
    else if (useConst && !useShared) convComposed<true, false><<<g, b>>>(din.data(), dfilter.data(), dout.data(), SZ, SZ);
    else convComposed<true, true><<<g, b>>>(din.data(), dfilter.data(), dout.data(), SZ, SZ);
    CHECK_CUDA(cudaDeviceSynchronize());
    std::vector<float> out(sz);
    dout.copyToHost(out.data());
    return out;
}

TEST(ConvTest, SeparableMatchesCpu) {
    std::vector<float> in, f; auto r = makeRef(in, f);
    size_t sz = in.size();
    DeviceBuffer<float> din(sz), dtmp(sz), dout(sz);
    din.copyFromHost(in.data());
    std::vector<float> hsep(FDIM, 1.0f / FDIM);
    CHECK_CUDA(cudaMemcpyToSymbol(cSep, hsep.data(), FDIM * sizeof(float)));
    dim3 b(CTILE, CTILE), g((SZ + CTILE - 1) / CTILE, (SZ + CTILE - 1) / CTILE);
    convSepH<<<g, b>>>(din.data(), dtmp.data(), SZ, SZ);
    convSepV<<<g, b>>>(dtmp.data(), dout.data(), SZ, SZ);
    CHECK_CUDA(cudaDeviceSynchronize());
    std::vector<float> out(sz); dout.copyToHost(out.data());
    EXPECT_TRUE(verifyApprox(out, r, 1e-3).ok);   // 분리 box == 2D box
}

TEST(ConvTest, NaiveMatchesCpu)    { std::vector<float> in, f; auto r = makeRef(in, f); EXPECT_TRUE(verifyApprox(run(false, false, in, f), r, 1e-4).ok); }
TEST(ConvTest, ConstantMatchesCpu) { std::vector<float> in, f; auto r = makeRef(in, f); EXPECT_TRUE(verifyApprox(run(true,  false, in, f), r, 1e-4).ok); }
TEST(ConvTest, SharedMatchesCpu)   { std::vector<float> in, f; auto r = makeRef(in, f); EXPECT_TRUE(verifyApprox(run(true,  true,  in, f), r, 1e-4).ok); }
