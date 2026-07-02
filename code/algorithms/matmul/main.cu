// main.cu
// compute-bound 케이스 스터디. harness 대신 자체 루프에서 GFLOP/s 측정 + verifyApprox (FP16 허용오차).
#include "../common/raii.cuh"
#include "../common/verify.cuh"
#include "../common/cli.cuh"
#include "matmul_kernels.cuh"

#include <cstdio>
#include <string>
#include <vector>

static void cpuMatmul(const std::vector<float>& A, const std::vector<float>& B,
                      std::vector<float>& C, int M, int N, int K) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float s = 0.0f;
            for (int k = 0; k < K; ++k) s += A[i * K + k] * B[k * N + j];
            C[i * N + j] = s;
        }
}

int main(int argc, char** argv) {
    BenchOptions opt = CliParser::parse(argc, argv);
    if (!opt.valid)   { CliParser::printUsage(argv[0]); return 1; }
    if (opt.showHelp) { CliParser::printUsage(argv[0]); return 0; }

    int Nsz = (opt.n > 0 && opt.n <= 8192) ? static_cast<int>(opt.n) : 512;
    Nsz = (Nsz / 16) * 16;                 // WMMA는 16의 배수
    if (Nsz < 16) Nsz = 16;
    int M = Nsz, N = Nsz, K = Nsz;
    int iters = opt.iters;

    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    std::printf("=== MatMul 벤치마크 (%dx%dx%d, %d회) ===\n", M, N, K, iters);
    std::printf("GPU : %s (SM %d.%d)\n\n", prop.name, prop.major, prop.minor);

    size_t szA = (size_t)M * K, szB = (size_t)K * N, szC = (size_t)M * N;
    std::vector<float> hA(szA), hB(szB), ref(szC), hOut(szC);
    for (size_t i = 0; i < szA; ++i) hA[i] = ((i % 7) - 3) * 0.1f;
    for (size_t i = 0; i < szB; ++i) hB[i] = ((i % 5) - 2) * 0.1f;
    cpuMatmul(hA, hB, ref, M, N, K);

    std::vector<half> hAh(szA), hBh(szB);
    for (size_t i = 0; i < szA; ++i) hAh[i] = __float2half(hA[i]);
    for (size_t i = 0; i < szB; ++i) hBh[i] = __float2half(hB[i]);

    DeviceBuffer<float> dA(szA), dB(szB), dC(szC);
    DeviceBuffer<half>  dAh(szA), dBh(szB);
    dA.copyFromHost(hA.data());   dB.copyFromHost(hB.data());
    dAh.copyFromHost(hAh.data()); dBh.copyFromHost(hBh.data());

    double flops = 2.0 * M * N * K;
    double tol   = 3.0e-2;                  // FP16 WMMA 허용오차

    std::printf("%-22s %10s %12s %8s %10s\n", "Stage", "Time(ms)", "GFLOP/s", "Check", "Speedup");
    std::printf("%s\n", std::string(66, '-').c_str());

    double baseMs = -1.0;
    auto bench = [&](const char* name, auto launch) {
        launch();
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        dC.copyToHost(hOut.data());
        auto vr = verifyApprox(hOut, ref, tol);

        GpuTimer t;
        t.start();
        for (int i = 0; i < iters; ++i) launch();
        double ms = t.stop() / iters;

        double gf = flops / (ms / 1.0e3) / 1.0e9;
        if (baseMs < 0) baseMs = ms;
        std::printf("%-22s %10.4f %12.1f %8s %9.2fx\n",
                    name, ms, gf, vr.ok ? "OK" : "FAIL", baseMs / ms);
    };

    dim3 b0(16, 16), g0((N + 15) / 16, (M + 15) / 16);
    bench("L0 naive", [&] { matmulComposed<false><<<g0, b0>>>(dA.data(), dB.data(), dC.data(), M, N, K); });
    bench("L1 tiled (shared)", [&] { matmulComposed<true><<<g0, b0>>>(dA.data(), dB.data(), dC.data(), M, N, K); });

    // L2 (고도화): register-tiled, 64x64 타일 블록당 256스레드(스레드당 4x4)
    dim3 gr((N + 63) / 64, (M + 63) / 64);
    bench("L2 register-tiled", [&] { sgemmRegTiled<64, 64, 8, 4, 4><<<gr, 256>>>(dA.data(), dB.data(), dC.data(), M, N, K); });

    // L3 (고도화): L2 + cp.async 더블버퍼 프리페치 (전역 로드 지연 은닉)
    bench("L3 +cp.async dbuf", [&] { sgemmRegTiledAsync<64, 64, 8, 4, 4><<<gr, 256>>>(dA.data(), dB.data(), dC.data(), M, N, K); });

    dim3 bw(128, 4), gw((M + 63) / 64, (N + 63) / 64);   // 워프 4x4 → 64x64 타일/블록
    bench("L4 WMMA (__half TC)", [&] { kWmma<<<gw, bw>>>(dAh.data(), dBh.data(), dC.data(), M, N, K); });

    // 참고: 1D-flatten 인덱싱(div/mod) vs 위 2D 인덱싱 비교 (L0와 알고리즘 동일, 매핑만 다름)
    bench("cf. naive 1D-index", [&] {
        int th = M * N;
        matmulNaive1D<<<(th + 255) / 256, 256>>>(dA.data(), dB.data(), dC.data(), M, N, K);
    });
    return 0;
}
