// main.cu
// 슬라이드: part5/chapter38 (Scan / Prefix Sum)
// scan 벤치마크 진입점. 코어 cli/harness/verify 재사용.
#include "../common/cli.cuh"
#include "../common/harness.cuh"
#include "../common/verify.cuh"
#include "../common/occupancy.cuh"
#include "scan_registry.cuh"
#include "scan_metric.cuh"

#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    auto all = makeScanVariants();
    BenchOptions opt = CliParser::parse(argc, argv);

    if (!opt.valid)   { CliParser::printUsage(argv[0]); return 1; }
    if (opt.showHelp) { CliParser::printUsage(argv[0]); return 0; }
    if (opt.listOnly) {
        std::printf("변형 목록:\n");
        for (const auto& v : all) std::printf("  %-9s %s\n", v.key.c_str(), v.label.c_str());
        return 0;
    }

    std::vector<Variant<ScanSig>> sel;
    if (opt.algoKeys.empty()) {
        sel = all;
    } else {
        for (const auto& k : opt.algoKeys) {
            bool found = false;
            for (const auto& v : all) if (v.key == k) { sel.push_back(v); found = true; break; }
            if (!found) { std::fprintf(stderr, "unknown algo key: %s (use -l)\n", k.c_str()); return 1; }
        }
    }

    int numSeg = static_cast<int>(opt.n / SEG);
    if (numSeg < 1) numSeg = 1;
    int N = numSeg * SEG;

    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    std::printf("=== Scan 벤치마크 ===\n");
    std::printf("GPU : %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    std::printf("N   : %d (%.1f MB), 세그먼트 %d x %d, %d회 평균\n\n",
                N, N * sizeof(int) / 1.0e6, numSeg, SEG, opt.iters);

    auto h_in = scanMakeInput(N);
    auto ref  = scanCpuReference(h_in, numSeg);

    runBenchmark<int, ScanSig>(
        sel, h_in, static_cast<size_t>(N), scanMetric(), static_cast<long>(N), opt.iters,
        [numSeg](const Variant<ScanSig>& v, const int* in, int* out) { v.launch(in, out, numSeg); },
        [&ref](const std::vector<int>& out) { return verifyExact(out, ref); });

    std::printf("\n이론 occupancy (blockDim=%d):\n", SEG);
    std::printf("  L1 Hillis-Steele : %5.0f%%\n", theoreticalOccupancy(scanComposed<false>, SEG));
    std::printf("  L2 warp-shuffle  : %5.0f%%\n", theoreticalOccupancy(scanComposed<true>, SEG));
    return 0;
}
