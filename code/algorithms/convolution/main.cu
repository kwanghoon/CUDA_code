// main.cu
// 슬라이드: part5/chapter35 (Convolution) — 자체 루프(shape 다름), GFLOP/s + verifyApprox.
#include "../common/raii.cuh"
#include "../common/verify.cuh"
#include "../common/cli.cuh"
#include "convolution_kernels.cuh"

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

static void cpuConv(const std::vector<float>& in, const std::vector<float>& f,
                    std::vector<float>& out, int W, int H) {
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            float s = 0.0f;
            for (int fy = -RADIUS; fy <= RADIUS; ++fy)
                for (int fx = -RADIUS; fx <= RADIUS; ++fx) {
                    int ix = std::min(std::max(x + fx, 0), W - 1);
                    int iy = std::min(std::max(y + fy, 0), H - 1);
                    s += in[iy * W + ix] * f[(fy + RADIUS) * FDIM + (fx + RADIUS)];
                }
            out[y * W + x] = s;
        }
}

int main(int argc, char** argv) {
    BenchOptions opt = CliParser::parse(argc, argv);
    if (!opt.valid)   { CliParser::printUsage(argv[0]); return 1; }
    if (opt.showHelp) { CliParser::printUsage(argv[0]); return 0; }

    int W = (opt.n > 0 && opt.n <= 16384) ? static_cast<int>(opt.n) : 2048;
    int H = W;
    int iters = opt.iters;

    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    std::printf("=== Convolution 벤치마크 (%dx%d, %dx%d 필터, %d회) ===\n", W, H, FDIM, FDIM, iters);
    std::printf("GPU : %s (SM %d.%d)\n\n", prop.name, prop.major, prop.minor);

    size_t sz = (size_t)W * H;
    std::vector<float> hin(sz), hout(sz), ref(sz), hf(FSIZE);
    for (size_t i = 0; i < sz; ++i) hin[i] = ((i % 13) - 6) * 0.1f;
    for (int i = 0; i < FSIZE; ++i) hf[i] = 1.0f / FSIZE;      // box blur
    cpuConv(hin, hf, ref, W, H);

    DeviceBuffer<float> din(sz), dout(sz), dfilter(FSIZE);
    din.copyFromHost(hin.data());
    dfilter.copyFromHost(hf.data());
    CHECK_CUDA(cudaMemcpyToSymbol(cFilter, hf.data(), FSIZE * sizeof(float)));
    std::vector<float> hsep(FDIM, 1.0f / FDIM);                 // 분리 필터 (box: 1/FDIM)
    CHECK_CUDA(cudaMemcpyToSymbol(cSep, hsep.data(), FDIM * sizeof(float)));
    DeviceBuffer<float> dtmp(sz);                               // separable 중간 버퍼

    double flops = 2.0 * sz * FSIZE;
    double tol   = 1.0e-4;

    std::printf("%-26s %10s %12s %8s %10s\n", "Stage", "Time(ms)", "GFLOP/s", "Check", "Speedup");
    std::printf("%s\n", std::string(70, '-').c_str());

    double baseMs = -1.0;
    auto bench = [&](const char* name, auto launch) {
        launch();
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        dout.copyToHost(hout.data());
        auto vr = verifyApprox(hout, ref, tol);
        GpuTimer t;
        t.start();
        for (int i = 0; i < iters; ++i) launch();
        double ms = t.stop() / iters;
        double gf = flops / (ms / 1.0e3) / 1.0e9;
        if (baseMs < 0) baseMs = ms;
        std::printf("%-26s %10.4f %12.1f %8s %9.2fx\n",
                    name, ms, gf, vr.ok ? "OK" : "FAIL", baseMs / ms);
    };

    dim3 b(CTILE, CTILE), g((W + CTILE - 1) / CTILE, (H + CTILE - 1) / CTILE);
    bench("L0 naive (global filter)", [&] { convComposed<false, false><<<g, b>>>(din.data(), dfilter.data(), dout.data(), W, H); });
    bench("L1 +constant filter",      [&] { convComposed<true,  false><<<g, b>>>(din.data(), dfilter.data(), dout.data(), W, H); });
    bench("L2 +shared tile (halo)",   [&] { convComposed<true,  true ><<<g, b>>>(din.data(), dfilter.data(), dout.data(), W, H); });
    bench("L3 separable (2x 1D)",     [&] {   // 고도화: 곱셈 25→10
        convSepH<<<g, b>>>(din.data(), dtmp.data(), W, H);
        convSepV<<<g, b>>>(dtmp.data(), dout.data(), W, H);
    });

    // alt (기법 데모): __grid_constant__ 필터 by-value (cudaMemcpyToSymbol 불필요)
    bench("alt __grid_constant__ filter", [&] {
        BoxFilter bf; for (int i = 0; i < FSIZE; ++i) bf.w[i] = hf[i];
        convGridConstFilter<<<g, b>>>(bf, din.data(), dout.data(), W, H);
    });

    // alt (compute ILP): register-tiled, 스레드당 TO×TO 독립 누산기 (FMA 파이프라인 채움)
    {
        constexpr int TO = 2, OT = 16 * TO, ST = OT + 2 * RADIUS;
        dim3 bi(16, 16), gi((W + OT - 1) / OT, (H + OT - 1) / OT);
        size_t smem = (size_t)ST * ST * sizeof(float);
        bench("alt reg-tiled ILP (2x2/thread)", [&] {
            convRegTileILP<TO><<<gi, bi, smem>>>(din.data(), dout.data(), W, H);
        });
    }

    // alt (기법 데모): 큰 동적 shared 타일 (>48KB opt-in) + L1/shared carveout
    {
        constexpr int BTILE = 128;
        constexpr int TW = BTILE + 2 * RADIUS;
        size_t smem = (size_t)TW * TW * sizeof(float);          // ~68KB > 48KB → 옵트인 필요
        cudaFuncSetAttribute(convBigTile<BTILE>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        cudaFuncSetAttribute(convBigTile<BTILE>, cudaFuncAttributePreferredSharedMemoryCarveout, 100); // shared 최대 선호
        dim3 bb(32, 32), gb((W + BTILE - 1) / BTILE, (H + BTILE - 1) / BTILE);
        bench("alt big dyn-smem tile (68KB)", [&] {
            convBigTile<BTILE><<<gb, bb, smem>>>(din.data(), dout.data(), W, H);
        });
    }

    // L2-persist (고도화): 재사용 큰 입력(din)을 L2 지속 캐시로 고정 → halo 재접근 히트↑.
    //   cudaAccessPolicyWindow(hitProp=Persisting)로 스트림에 접근정책 창을 건다.
    //   Orin은 L2가 커서 이미 잘 캐싱 → 이득이 작을 수 있어 'cond.'로 표기(측정이 판단).
    if (prop.persistingL2CacheMaxSize > 0) {
        size_t winBytes = std::min({sz * sizeof(float),
                                    (size_t)prop.persistingL2CacheMaxSize,
                                    (size_t)prop.accessPolicyMaxWindowSize});
        CHECK_CUDA(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, winBytes));
        cudaStream_t st; CHECK_CUDA(cudaStreamCreate(&st));
        cudaStreamAttrValue av{};
        av.accessPolicyWindow.base_ptr  = din.data();
        av.accessPolicyWindow.num_bytes = winBytes;
        av.accessPolicyWindow.hitRatio  = 1.0f;
        av.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;   // 창 내부 = L2 지속
        av.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;    // 창 외부 = 스트리밍
        CHECK_CUDA(cudaStreamSetAttribute(st, cudaStreamAttributeAccessPolicyWindow, &av));

        auto launch = [&] { convComposed<true, true><<<g, b, 0, st>>>(din.data(), dfilter.data(), dout.data(), W, H); };
        launch(); CHECK_CUDA(cudaStreamSynchronize(st));
        dout.copyToHost(hout.data());
        auto vr = verifyApprox(hout, ref, tol);
        GpuTimer t; t.start(st);
        for (int i = 0; i < iters; ++i) launch();
        double msec = t.stop(st) / iters;
        std::printf("%-26s %10.4f %12.1f %8s %9.2fx\n", "L2-persist shared (cond.)",
                    msec, flops / (msec / 1.0e3) / 1.0e9, vr.ok ? "OK" : "FAIL", baseMs / msec);

        av.accessPolicyWindow.num_bytes = 0;
        cudaStreamSetAttribute(st, cudaStreamAttributeAccessPolicyWindow, &av);
        CHECK_CUDA(cudaCtxResetPersistingL2Cache());
        cudaStreamDestroy(st);
    } else {
        std::printf("%-26s %10s  persistingL2 미지원 → 건너뜀\n", "L2-persist shared", "-");
    }
    return 0;
}
