// main.cu
// 슬라이드: part5/chapter40 (SpMV) — 자체 루프, GFLOP/s + verifyApprox. CSR, 행당 K 비영.
#include "../common/raii.cuh"
#include "../common/verify.cuh"
#include "../common/cli.cuh"
#include "spmv_kernels.cuh"

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    BenchOptions opt = CliParser::parse(argc, argv);
    if (!opt.valid)   { CliParser::printUsage(argv[0]); return 1; }
    if (opt.showHelp) { CliParser::printUsage(argv[0]); return 0; }

    int  M = (opt.n > 0 && opt.n <= (1 << 22)) ? static_cast<int>(opt.n) : 65536;
    int  N = M;
    int  K = 32;                       // 행당 비영 개수
    long nnz = (long)M * K;
    int  iters = opt.iters;

    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    std::printf("=== SpMV 벤치마크 (CSR, M=%d, nnz=%ld, K=%d/행, %d회) ===\n", M, nnz, K, iters);
    std::printf("GPU : %s (SM %d.%d)\n\n", prop.name, prop.major, prop.minor);

    std::vector<int> rowPtr(M + 1), colIdx(nnz);
    std::vector<float> vals(nnz), x(N), ref(M);
    for (int r = 0; r <= M; ++r) rowPtr[r] = r * K;
    unsigned s = 42u;
    for (long i = 0; i < nnz; ++i) {
        s = s * 1103515245u + 12345u; colIdx[i] = (int)((s >> 9) % N);
        vals[i] = (((s >> 16) & 0xff) / 255.0f) - 0.5f;
    }
    for (int i = 0; i < N; ++i) { s = s * 1103515245u + 12345u; x[i] = (((s >> 16) & 0xff) / 255.0f) - 0.5f; }
    for (int r = 0; r < M; ++r) {
        float acc = 0.0f;
        for (int k = rowPtr[r]; k < rowPtr[r + 1]; ++k) acc += vals[k] * x[colIdx[k]];
        ref[r] = acc;
    }

    DeviceBuffer<int>   dRow(M + 1), dCol(nnz);
    DeviceBuffer<float> dVal(nnz), dX(N), dY(M);
    dRow.copyFromHost(rowPtr.data()); dCol.copyFromHost(colIdx.data());
    dVal.copyFromHost(vals.data());   dX.copyFromHost(x.data());

    double flops = 2.0 * nnz;
    double tol   = 1.0e-2;

    std::printf("%-24s %10s %12s %8s %10s\n", "Stage", "Time(ms)", "GFLOP/s", "Check", "Speedup");
    std::printf("%s\n", std::string(68, '-').c_str());

    std::vector<float> hy(M);
    double baseMs = -1.0;
    auto bench = [&](const char* name, auto launch) {
        launch();
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        dY.copyToHost(hy.data());
        auto vr = verifyApprox(hy, ref, tol);
        GpuTimer t;
        t.start();
        for (int i = 0; i < iters; ++i) launch();
        double ms = t.stop() / iters;
        double gf = flops / (ms / 1.0e3) / 1.0e9;
        if (baseMs < 0) baseMs = ms;
        std::printf("%-24s %10.4f %12.2f %8s %9.2fx\n",
                    name, ms, gf, vr.ok ? "OK" : "FAIL", baseMs / ms);
    };

    bench("L0 scalar (thread/row)", [&] {
        spmvScalar<<<(M + 255) / 256, 256>>>(dRow.data(), dCol.data(), dVal.data(), dX.data(), dY.data(), M);
    });
    bench("L1 vector (warp/row)", [&] {
        long th = (long)M * 32;
        spmvVector<false><<<(int)((th + 127) / 128), 128>>>(dRow.data(), dCol.data(), dVal.data(), dX.data(), dY.data(), M);
    });
    bench("L2 vector +__ldg", [&] {
        long th = (long)M * 32;
        spmvVector<true><<<(int)((th + 127) / 128), 128>>>(dRow.data(), dCol.data(), dVal.data(), dX.data(), dY.data(), M);
    });

    // L3 ELL: CSR → 열-major 패딩 포맷으로 변환 (maxNnz = 최대 행길이).
    int maxNnz = 0;
    for (int r = 0; r < M; ++r) maxNnz = std::max(maxNnz, rowPtr[r + 1] - rowPtr[r]);
    std::vector<int>   ellCol((long)maxNnz * M, -1);
    std::vector<float> ellVal((long)maxNnz * M, 0.0f);
    for (int r = 0; r < M; ++r) {
        int j = 0;
        for (int k = rowPtr[r]; k < rowPtr[r + 1]; ++k, ++j) {
            ellCol[(long)j * M + r] = colIdx[k];
            ellVal[(long)j * M + r] = vals[k];
        }
    }
    DeviceBuffer<int>   dEllCol((long)maxNnz * M);
    DeviceBuffer<float> dEllVal((long)maxNnz * M);
    dEllCol.copyFromHost(ellCol.data());
    dEllVal.copyFromHost(ellVal.data());
    bench("L3 ELL (coalesced rows)", [&] {
        spmvEll<<<(M + 255) / 256, 256>>>(dEllCol.data(), dEllVal.data(), dX.data(), dY.data(), M, maxNnz);
    });

    // AoS vs SoA 케이스: (val,col)을 인터리브한 AoS. L1/L2(SoA) 와 비교.
    //   두 필드를 함께 쓰므로 AoS도 coalesce well → SoA 대비 차이 작음(정직히 표기).
    std::vector<SpElem> hAoS(nnz);
    for (long k = 0; k < nnz; ++k) { hAoS[k].val = vals[k]; hAoS[k].col = colIdx[k]; }
    DeviceBuffer<SpElem> dAoS(nnz);
    dAoS.copyFromHost(hAoS.data());
    bench("AoS (val,col interleaved)", [&] {
        long th = (long)M * 32;
        spmvVectorAoS<<<(int)((th + 127) / 128), 128>>>(dRow.data(), dAoS.data(), dX.data(), dY.data(), M);
    });
    return 0;
}
