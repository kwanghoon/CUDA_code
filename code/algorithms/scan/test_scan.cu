// test_scan.cu
// 슬라이드: part5/chapter38 (Scan / Prefix Sum)
// GoogleTest: 레지스트리의 모든 변형을 CPU 기준과 비교 (파라미터라이즈드).
#include <gtest/gtest.h>

#include "../common/raii.cuh"
#include "../common/variant.cuh"
#include "../common/verify.cuh"
#include "scan_registry.cuh"
#include "scan_metric.cuh"

#include <functional>
#include <vector>

static std::vector<int> runOnce(const std::function<ScanSig>& launch,
                                const std::vector<int>& in, int numSeg) {
    int N = static_cast<int>(in.size());
    DeviceBuffer<int> d_in(N), d_out(N);
    d_in.copyFromHost(in.data());
    launch(d_in.data(), d_out.data(), numSeg);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    std::vector<int> out(N);
    d_out.copyToHost(out.data());
    return out;
}

class ScanVariantTest : public ::testing::TestWithParam<Variant<ScanSig>> {};

TEST_P(ScanVariantTest, MatchesCpuReference) {
    const int numSeg = 1000, N = numSeg * SEG;
    auto in  = scanMakeInput(N);
    auto ref = scanCpuReference(in, numSeg);
    auto out = runOnce(GetParam().launch, in, numSeg);
    EXPECT_TRUE(verifyExact(out, ref).ok);
}

TEST_P(ScanVariantTest, SingleSegment) {
    const int numSeg = 1, N = SEG;
    auto in  = scanMakeInput(N);
    auto ref = scanCpuReference(in, numSeg);
    auto out = runOnce(GetParam().launch, in, numSeg);
    EXPECT_TRUE(verifyExact(out, ref).ok);
}

INSTANTIATE_TEST_SUITE_P(
    AllScanVariants, ScanVariantTest,
    ::testing::ValuesIn(makeScanVariants()),
    [](const testing::TestParamInfo<Variant<ScanSig>>& info) { return info.param.key; });
