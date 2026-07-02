// test_histogram.cu
// 슬라이드: part5/chapter37 (Histogram)
// GoogleTest: 모든 히스토그램 레벨을 CPU 기준과 비교 (파라미터라이즈드).
#include <gtest/gtest.h>

#include "../common/raii.cuh"
#include "../common/variant.cuh"
#include "../common/verify.cuh"
#include "histogram_registry.cuh"
#include "histogram_metric.cuh"

#include <functional>
#include <vector>

static std::vector<int> runOnce(const std::function<HistSig>& launch, const std::vector<int>& in) {
    int N = static_cast<int>(in.size());
    DeviceBuffer<int> d_in(N), d_out(NBINS);
    d_in.copyFromHost(in.data());
    launch(d_in.data(), d_out.data(), N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    std::vector<int> out(NBINS);
    d_out.copyToHost(out.data());
    return out;
}

class HistVariantTest : public ::testing::TestWithParam<Variant<HistSig>> {};

TEST_P(HistVariantTest, MatchesCpuHistogram) {
    auto in  = histMakeInput(1 << 20);
    auto ref = histCpuReference(in);
    auto out = runOnce(GetParam().launch, in);
    EXPECT_TRUE(verifyExact(out, ref).ok);
}

INSTANTIATE_TEST_SUITE_P(
    AllHistVariants, HistVariantTest,
    ::testing::ValuesIn(makeHistogramVariants()),
    [](const testing::TestParamInfo<Variant<HistSig>>& info) { return info.param.key; });
