// test_sort.cu
// 슬라이드: part5/chapter36 (Sorting)
// GoogleTest: bitonic 레벨을 std::sort 기준과 비교 (파라미터라이즈드).
#include <gtest/gtest.h>

#include "../common/raii.cuh"
#include "../common/variant.cuh"
#include "../common/verify.cuh"
#include "sort_registry.cuh"
#include "sort_metric.cuh"

#include <functional>
#include <vector>

static std::vector<int> runOnce(const std::function<SortSig>& launch, const std::vector<int>& in) {
    int N = static_cast<int>(in.size());
    DeviceBuffer<int> d_in(N), d_out(N);
    d_in.copyFromHost(in.data());
    launch(d_in.data(), d_out.data(), N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    std::vector<int> out(N);
    d_out.copyToHost(out.data());
    return out;
}

class SortVariantTest : public ::testing::TestWithParam<Variant<SortSig>> {};

TEST_P(SortVariantTest, MatchesStdSort) {
    auto in  = sortMakeInput(1 << 12);      // 4096 (2^k)
    auto ref = sortCpuReference(in);
    auto out = runOnce(GetParam().launch, in);
    EXPECT_TRUE(verifyExact(out, ref).ok);
}

INSTANTIATE_TEST_SUITE_P(
    AllSortVariants, SortVariantTest,
    ::testing::ValuesIn(makeSortVariants()),
    [](const testing::TestParamInfo<Variant<SortSig>>& info) { return info.param.key; });
