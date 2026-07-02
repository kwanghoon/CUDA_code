// 실습: Matrix Multiply 최적화 — BASELINE (여기서 시작)
// 목표: naive matmul 을 ncu로 병목을 찾아 tiling → register-tiling(ILP) 순으로 최적화.
// matmul은 COMPUTE-bound → 대역폭이 아니라 연산 활용/재사용이 관건.
//   빌드:  nvcc -O3 -arch=sm_87 baseline.cu -o baseline
//   프로파일: ncu --set full ./baseline
//   STEP 1  l1tex__t_sector_hit_rate.pct 낮음 / dram__bytes.sum 큼 → shared memory TILING
//   STEP 2  sm__throughput.avg.pct_of_peak → REGISTER TILING (스레드당 TM×TN 레지스터 누산 = ILP)
//   STEP 3  sm__pipe_fp32_cycles_active / dram__throughput → AI 로 roofline 위치 확인
//   matmul은 compute-bound: sm__throughput(FP32)을 peak 쪽으로. (더 = __half+WMMA, 심화)
#include "../../common/raii.cuh"
#include <cstdio>
#include <vector>

// BASELINE: naive — 스레드당 C[r][c] 하나, A행·B열을 매번 전역에서 읽음 (재사용 0)
__global__ void matmulBaseline(const float* __restrict__ A, const float* __restrict__ B,
                               float* __restrict__ C, int N) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < N && c < N) {
        float s = 0.0f;
        for (int k = 0; k < N; ++k) s += A[r * N + k] * B[k * N + c];   // 전역 반복 로드
        C[r * N + c] = s;
    }
}

int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 1024;
    std::vector<float> A(N * N), B(N * N), C(N * N);
    for (int i = 0; i < N * N; ++i) { A[i] = (i % 7) * 0.1f; B[i] = (i % 5) * 0.1f; }

    DeviceBuffer<float> dA(N * N), dB(N * N), dC(N * N);
    dA.copyFromHost(A.data()); dB.copyFromHost(B.data());

    dim3 block(16, 16), grid((N + 15) / 16, (N + 15) / 16);
    matmulBaseline<<<grid, block>>>(dA.data(), dB.data(), dC.data(), N);
    CHECK_CUDA(cudaDeviceSynchronize());

    GpuTimer t; t.start();
    const int iters = 30;
    for (int k = 0; k < iters; ++k) matmulBaseline<<<grid, block>>>(dA.data(), dB.data(), dC.data(), N);
    double ms = t.stop() / iters;

    dC.copyToHost(C.data());
    int rr = N / 3, cc = N / 2; double ref = 0;
    for (int k = 0; k < N; ++k) ref += (double)A[rr * N + k] * B[k * N + cc];
    bool ok = std::abs(C[rr * N + cc] - ref) < 1e-2 * (1 + std::abs(ref));

    double gflops = 2.0 * N * N * N / (ms / 1e3) / 1e9;
    std::printf("baseline: %.4f ms   %.1f GFLOP/s   check=%s\n", ms, gflops, ok ? "OK" : "FAIL");
    std::printf("→ ncu --set full ./baseline 로 STEP 1~3(재사용/컴퓨트/roofline) 을 보고 tiling부터 적용하라.\n");
    return 0;
}
