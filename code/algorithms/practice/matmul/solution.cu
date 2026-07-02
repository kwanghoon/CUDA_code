// 실습: Matrix Multiply 최적화 — SOLUTION (완전 최적화 참고답안)
// baseline.cu 의 STEP 1~2 적용: shared tiling + register tiling(ILP).
//   l1tex__t_sector_hit_rate.pct → 상승 (shared 타일 재사용)
//   dram__bytes.sum → 급감 (전역 반복 로드 제거)
//   sm__throughput.avg.pct_of_peak_sustained → 상승 (register-tiling ILP)
//   sm__pipe_fp32_cycles_active.avg.pct → 높음 (compute-bound 상한)
// 적용 기법: 64×64 타일을 shared로, 스레드당 4×4 출력을 레지스터 누산(acc[4][4]=ILP).
// (chapter34 의 sgemmRegTiled<64,64,8,4,4> 와 동일)
#include "../../common/raii.cuh"
#include <cstdio>
#include <vector>

template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM / TM) * (BN / TN))
sgemmRegTiled(const float* __restrict__ A, const float* __restrict__ B,
              float* __restrict__ C, int M, int N, int K) {
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];
    int blockRow = blockIdx.y * BM, blockCol = blockIdx.x * BN;
    int tRow = threadIdx.x / (BN / TN);
    int tCol = threadIdx.x % (BN / TN);
    const int numThreads = (BM / TM) * (BN / TN);

    float acc[TM][TN];                              // 독립 누산기 = ILP
    #pragma unroll
    for (int i = 0; i < TM; ++i) for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;
    float regA[TM], regB[TN];

    for (int k0 = 0; k0 < K; k0 += BK) {
        for (int i = threadIdx.x; i < BM * BK; i += numThreads) {   // A 타일 → shared
            int r = i / BK, c = i % BK;
            As[i] = (blockRow + r < M && k0 + c < K) ? A[(blockRow + r) * K + k0 + c] : 0.0f;
        }
        for (int i = threadIdx.x; i < BK * BN; i += numThreads) {   // B 타일 → shared
            int r = i / BN, c = i % BN;
            Bs[i] = (k0 + r < K && blockCol + c < N) ? B[(k0 + r) * N + blockCol + c] : 0.0f;
        }
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            #pragma unroll
            for (int i = 0; i < TM; ++i) regA[i] = As[(tRow * TM + i) * BK + kk];
            #pragma unroll
            for (int j = 0; j < TN; ++j) regB[j] = Bs[kk * BN + tCol * TN + j];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                for (int j = 0; j < TN; ++j) acc[i][j] += regA[i] * regB[j];   // TM×TN 독립 FMA
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j) {
            int r = blockRow + tRow * TM + i, c = blockCol + tCol * TN + j;
            if (r < M && c < N) C[r * N + c] = acc[i][j];
        }
}

int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 1024;
    std::vector<float> A(N * N), B(N * N), C(N * N);
    for (int i = 0; i < N * N; ++i) { A[i] = (i % 7) * 0.1f; B[i] = (i % 5) * 0.1f; }

    DeviceBuffer<float> dA(N * N), dB(N * N), dC(N * N);
    dA.copyFromHost(A.data()); dB.copyFromHost(B.data());

    dim3 grid((N + 63) / 64, (N + 63) / 64);
    sgemmRegTiled<64, 64, 8, 4, 4><<<grid, 256>>>(dA.data(), dB.data(), dC.data(), N, N, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    GpuTimer t; t.start();
    const int iters = 30;
    for (int k = 0; k < iters; ++k) sgemmRegTiled<64, 64, 8, 4, 4><<<grid, 256>>>(dA.data(), dB.data(), dC.data(), N, N, N);
    double ms = t.stop() / iters;

    dC.copyToHost(C.data());
    int rr = N / 3, cc = N / 2; double ref = 0;
    for (int k = 0; k < N; ++k) ref += (double)A[rr * N + k] * B[k * N + cc];
    bool ok = std::abs(C[rr * N + cc] - ref) < 1e-2 * (1 + std::abs(ref));

    double gflops = 2.0 * N * N * N / (ms / 1e3) / 1e9;
    std::printf("solution: %.4f ms   %.1f GFLOP/s   check=%s\n", ms, gflops, ok ? "OK" : "FAIL");
    return 0;
}
