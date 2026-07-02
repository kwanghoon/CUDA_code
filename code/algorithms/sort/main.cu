// main.cu
// 슬라이드: part5/chapter36 (Sorting) — bitonic 벤치. N은 2의 거듭제곱으로 내림.
#include "../common/cli.cuh"
#include "../common/harness.cuh"
#include "../common/verify.cuh"
#include "sort_registry.cuh"
#include "sort_metric.cuh"

#include <cstdio>
#include <string>
#include <vector>

static int pow2floor(long x) { int p = 1; while ((long)p * 2 <= x) p <<= 1; return p; }

int main(int argc, char** argv) {
    auto all = makeSortVariants();
    BenchOptions opt = CliParser::parse(argc, argv);
    if (!opt.valid)   { CliParser::printUsage(argv[0]); return 1; }
    if (opt.showHelp) { CliParser::printUsage(argv[0]); return 0; }
    if (opt.listOnly) {
        std::printf("변형 목록:\n");
        for (const auto& v : all) std::printf("  %-4s %s\n", v.key.c_str(), v.label.c_str());
        return 0;
    }

    std::vector<Variant<SortSig>> sel;
    if (opt.algoKeys.empty()) sel = all;
    else for (const auto& k : opt.algoKeys) {
        bool found = false;
        for (const auto& v : all) if (v.key == k) { sel.push_back(v); found = true; break; }
        if (!found) { std::fprintf(stderr, "unknown algo key: %s (use -l)\n", k.c_str()); return 1; }
    }

    int N = pow2floor(opt.n > 0 ? opt.n : (1L << 20));

    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    std::printf("=== Bitonic Sort 벤치마크 (N=%d, %d회) ===\n", N, opt.iters);
    std::printf("GPU : %s (SM %d.%d)   (GB/s는 다중패스 근사)\n\n", prop.name, prop.major, prop.minor);

    auto h_in = sortMakeInput(N);
    auto ref  = sortCpuReference(h_in);

    runBenchmark<int, SortSig>(
        sel, h_in, static_cast<size_t>(N), sortMetric(), static_cast<long>(N), opt.iters,
        [N](const Variant<SortSig>& v, const int* in, int* out) { v.launch(in, out, N); },
        [&ref](const std::vector<int>& out) { return verifyExact(out, ref); });
    return 0;
}
