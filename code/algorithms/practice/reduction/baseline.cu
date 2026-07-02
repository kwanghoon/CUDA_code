// 실습: Reduction 최적화 — BASELINE (여기서 시작해서 직접 최적화하라)
// 목표: naive 리덕션을 ncu로 병목을 찾아 한 기법씩 적용해 solution.cu 수준까지.
//   빌드:  nvcc -O3 -arch=sm_87 baseline.cu -o baseline
//   프로파일: ncu --set full ./baseline  (특정 metric: ncu --metrics <name> ./baseline)
//   STEP 1  smsp__thread_inst_executed_per_inst_executed_realtime → sequential addressing (divergence 해소)
//   STEP 2  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum → bank conflict 해소 (필요시 +1 패딩)
//   STEP 3  smsp__inst_executed_op_barrier → warp-tail (s<=32 sync-free)
//   STEP 4  sm__warps_active.avg.pct_of_peak_sustained_active → grid-stride + SM 맞춤
//   STEP 5  dram__throughput.avg.pct_of_peak_sustained_elapsed (>80%) → int4 벡터화 로드
//   리덕션은 memory-bound: dram 처리량이 peak 근처면 최적.
#include "../../common/raii.cuh"
#include <cstdio>
#include <vector>

constexpr int BLOCK = 256;

// BASELINE: divergent interleaved-addressing 리덕션
__global__ void reduceBaseline(const int* __restrict__ in, int* __restrict__ out, int n) {
    __shared__ int s[BLOCK];
    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + tid;
    s[tid] = (i < n) ? in[i] : 0;
    __syncthreads();
    // STEP 1 병목: tid % (2*s) 분기 → 워프 divergence + bank conflict
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (tid % (2 * stride) == 0) s[tid] += s[tid + stride];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, s[0]);
}

int main(int argc, char** argv) {
    int n = (argc > 1) ? atoi(argv[1]) : (1 << 24);
    std::vector<int> h(n);
    long cpu = 0;
    for (int i = 0; i < n; ++i) { h[i] = (i % 7) - 3; cpu += h[i]; }

    DeviceBuffer<int> din(n), dout(1);
    din.copyFromHost(h.data());
    int zero = 0; dout.copyFromHost(&zero);

    int grid = (n + BLOCK - 1) / BLOCK;
    reduceBaseline<<<grid, BLOCK>>>(din.data(), dout.data(), n);
    CHECK_CUDA(cudaDeviceSynchronize());

    GpuTimer t; t.start();
    const int iters = 200;
    for (int k = 0; k < iters; ++k) {
        cudaMemset(dout.data(), 0, sizeof(int));
        reduceBaseline<<<grid, BLOCK>>>(din.data(), dout.data(), n);
    }
    double ms = t.stop() / iters;

    int got; dout.copyToHost(&got);
    double gbps = (double)n * sizeof(int) / (ms / 1e3) / 1e9;
    std::printf("baseline: %.4f ms   %.1f GB/s   check=%s\n",
                ms, gbps, (got == (int)cpu) ? "OK" : "FAIL");
    std::printf("→ ncu --set full ./baseline 로 위 STEP 1~5 metric을 보고 하나씩 고쳐라.\n");
    return 0;
}
