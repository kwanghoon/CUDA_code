// test_reduction.cu
// GoogleTest: 모든 리덕션 변형을 CPU 합계와 비교 (파라미터라이즈드).
#include <gtest/gtest.h>

#include "../common/raii.cuh"
#include "../common/variant.cuh"
#include "../common/verify.cuh"
#include "reduction_registry.cuh"
#include "reduction_metric.cuh"

#include <functional>
#include <vector>

static int runOnce(const std::function<ReduceSig>& launch, const std::vector<int>& in) {
    int N = static_cast<int>(in.size());
    DeviceBuffer<int> d_in(N), d_out(1);
    d_in.copyFromHost(in.data());
    launch(d_in.data(), d_out.data(), N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    int out = 0;
    d_out.copyToHost(&out);
    return out;
}

class ReduceVariantTest : public ::testing::TestWithParam<Variant<ReduceSig>> {};

TEST_P(ReduceVariantTest, MatchesCpuSum) {
    auto in  = reduceMakeInput(1 << 20);
    auto ref = reduceCpuReference(in);
    int out  = runOnce(GetParam().launch, in);
    EXPECT_EQ(out, ref[0]);
}

TEST_P(ReduceVariantTest, SmallInput) {
    auto in  = reduceMakeInput(1000);
    auto ref = reduceCpuReference(in);
    int out  = runOnce(GetParam().launch, in);
    EXPECT_EQ(out, ref[0]);
}

INSTANTIATE_TEST_SUITE_P(
    AllReduceVariants, ReduceVariantTest,
    ::testing::ValuesIn(makeReductionVariants()),
    [](const testing::TestParamInfo<Variant<ReduceSig>>& info) { return info.param.key; });
