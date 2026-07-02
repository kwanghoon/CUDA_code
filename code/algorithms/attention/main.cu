// main.cu
// 슬라이드: part10/chapter77 (Attention) — 자체 루프, GFLOP/s + verifyApprox.
#include "../common/raii.cuh"
#include "../common/verify.cuh"
#include "../common/cli.cuh"
#include "attention_kernels.cuh"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

static void cpuAttention(const std::vector<float>& Q, const std::vector<float>& K,
                         const std::vector<float>& V, std::vector<float>& O, int N) {
    std::vector<float> s(N);
    float scale = 1.0f / std::sqrt((float)HEAD_DIM);
    for (int i = 0; i < N; ++i) {
        float m = -1e30f;
        for (int j = 0; j < N; ++j) {
            float d = 0.0f;
            for (int k = 0; k < HEAD_DIM; ++k) d += Q[i * HEAD_DIM + k] * K[j * HEAD_DIM + k];
            s[j] = d * scale; m = std::fmax(m, s[j]);
        }
        float l = 0.0f;
        for (int j = 0; j < N; ++j) { s[j] = std::exp(s[j] - m); l += s[j]; }
        for (int t = 0; t < HEAD_DIM; ++t) {
            float acc = 0.0f;
            for (int j = 0; j < N; ++j) acc += s[j] * V[j * HEAD_DIM + t];
            O[i * HEAD_DIM + t] = acc / l;
        }
    }
}

// L2 전략: K(재사용)를 L2 persisting, 나머지는 streaming (per-tensor 캐시 전략).
static void setKVPersist(const float* K, size_t bytes) {
    cudaDeviceProp p{};
    cudaGetDeviceProperties(&p, 0);
    if (p.persistingL2CacheMaxSize == 0) return;
    cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, p.persistingL2CacheMaxSize);
    cudaStreamAttrValue attr{};
    attr.accessPolicyWindow.base_ptr  = const_cast<float*>(K);
    size_t maxW = p.accessPolicyMaxWindowSize;
    attr.accessPolicyWindow.num_bytes = bytes < maxW ? bytes : maxW;
    attr.accessPolicyWindow.hitRatio  = 1.0f;
    attr.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
    attr.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;
    cudaStreamSetAttribute(0, cudaStreamAttributeAccessPolicyWindow, &attr);
}

int main(int argc, char** argv) {
    BenchOptions opt = CliParser::parse(argc, argv);
    if (!opt.valid)   { CliParser::printUsage(argv[0]); return 1; }
    if (opt.showHelp) { CliParser::printUsage(argv[0]); return 0; }

    int N = (opt.n > 0 && opt.n <= 8192) ? static_cast<int>(opt.n) : 1024;  // seq len
    int iters = opt.iters;

    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    std::printf("=== Attention 벤치마크 (N=%d, head_dim=%d, %d회) ===\n", N, HEAD_DIM, iters);
    std::printf("GPU : %s (SM %d.%d)\n\n", prop.name, prop.major, prop.minor);

    size_t sz = (size_t)N * HEAD_DIM;
    std::vector<float> Q(sz), K(sz), Vv(sz), O(sz), ref(sz);
    unsigned s = 1234u;
    auto rnd = [&]() { s = s * 1103515245u + 12345u; return (((s >> 16) & 0xff) / 255.0f) - 0.5f; };
    for (size_t i = 0; i < sz; ++i) { Q[i] = rnd(); K[i] = rnd(); Vv[i] = rnd(); }
    cpuAttention(Q, K, Vv, ref, N);

    DeviceBuffer<float> dQ(sz), dK(sz), dV(sz), dO(sz);
    dQ.copyFromHost(Q.data()); dK.copyFromHost(K.data()); dV.copyFromHost(Vv.data());

    double flops = 4.0 * (double)N * N * HEAD_DIM;   // QK + PV
    float  tol   = 2.0e-2f;

    std::printf("%-24s %10s %12s %8s %10s\n", "Stage", "Time(ms)", "GFLOP/s", "Check", "Speedup");
    std::printf("%s\n", std::string(68, '-').c_str());

    double baseMs = -1.0;
    auto bench = [&](const char* name, auto launch) {
        launch();
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        dO.copyToHost(O.data());
        auto vr = verifyApprox(O, ref, tol);
        GpuTimer t;
        t.start();
        for (int i = 0; i < iters; ++i) launch();
        double ms = t.stop() / iters;
        double gf = flops / (ms / 1.0e3) / 1.0e9;
        if (baseMs < 0) baseMs = ms;
        std::printf("%-24s %10.4f %12.1f %8s %9.2fx\n",
                    name, ms, gf, vr.ok ? "OK" : "FAIL", baseMs / ms);
    };

    size_t shBytes = (size_t)(N + HEAD_DIM + HEAD_DIM) * sizeof(float);
    bench("L0 naive (scores[N])", [&] { attnNaive<<<N, HEAD_DIM, shBytes>>>(dQ.data(), dK.data(), dV.data(), dO.data(), N); });
    bench("L1 flash (online)",    [&] { attnFlash<false><<<N, HEAD_DIM>>>(dQ.data(), dK.data(), dV.data(), dO.data(), N); });
    bench("L2 flash +K/V L2-cache", [&] {
        setKVPersist(dK.data(), sz * sizeof(float));   // K 재사용 → L2 persisting, Q streaming(__ldcs)
        attnFlash<true><<<N, HEAD_DIM>>>(dQ.data(), dK.data(), dV.data(), dO.data(), N);
    });
    return 0;
}
