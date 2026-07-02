// main.cu
// 슬라이드: part5/chapter37 (Histogram)
// histogram 벤치마크 진입점. 코어 cli/harness/verify 재사용.
#include "../common/cli.cuh"
#include "../common/harness.cuh"
#include "../common/verify.cuh"
#include "histogram_registry.cuh"
#include "histogram_metric.cuh"

#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    auto all = makeHistogramVariants();
    BenchOptions opt = CliParser::parse(argc, argv);

    if (!opt.valid)   { CliParser::printUsage(argv[0]); return 1; }
    if (opt.showHelp) { CliParser::printUsage(argv[0]); return 0; }
    if (opt.listOnly) {
        std::printf("변형 목록:\n");
        for (const auto& v : all) std::printf("  %-4s %s\n", v.key.c_str(), v.label.c_str());
        return 0;
    }

    std::vector<Variant<HistSig>> sel;
    if (opt.algoKeys.empty()) {
        sel = all;
    } else {
        for (const auto& k : opt.algoKeys) {
            bool found = false;
            for (const auto& v : all) if (v.key == k) { sel.push_back(v); found = true; break; }
            if (!found) { std::fprintf(stderr, "unknown algo key: %s (use -l)\n", k.c_str()); return 1; }
        }
    }

    int N = static_cast<int>(opt.n);
    if (N < 1) N = 1;

    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    std::printf("=== Histogram 벤치마크 (bins=%d) ===\n", NBINS);
    std::printf("GPU : %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    std::printf("N   : %d (%.1f MB), %d회 평균\n\n", N, N * sizeof(int) / 1.0e6, opt.iters);

    auto h_in = histMakeInput(N);
    auto ref  = histCpuReference(h_in);

    runBenchmark<int, HistSig>(
        sel, h_in, static_cast<size_t>(NBINS), histMetric(), static_cast<long>(N), opt.iters,
        [N](const Variant<HistSig>& v, const int* in, int* out) { v.launch(in, out, N); },
        [&ref](const std::vector<int>& out) { return verifyExact(out, ref); });
    return 0;
}
