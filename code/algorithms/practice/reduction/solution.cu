// 실습: Reduction 최적화 — SOLUTION (완전 최적화 참고답안)
// baseline.cu 의 STEP 1~5 를 모두 적용한 결과.
//   smsp__thread_inst_executed_per_inst_executed_realtime → ~1.0 (divergence 해소)
//   l1tex__data_bank_conflicts_...shared_op_ld.sum → ~0 (sequential addressing)
//   smsp__inst_executed_op_barrier → 감소 (warp-tail)
//   sm__warps_active.avg.pct_of_peak_sustained_active → 상승 (grid-stride, SM 맞춤)
//   dram__throughput.avg.pct_of_peak_sustained_elapsed → >80% (int4 벡터화)
// 적용 기법(누적): sequential addressing → warp-tail(sync-free) → grid-stride → int4 벡터화.
// (chapter33 의 reduceComposed<...> 와 동일한 최종형)
#include "../../common/raii.cuh"
#include <cstdio>
#include <vector>

constexpr int BLOCK = 256;

__device__ __forceinline__ int warpReduceSum(int v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}

// SOLUTION: int4 grid-stride 로드 + sequential addressing + warp-tail
__global__ void reduceSolution(const int* __restrict__ in, int* __restrict__ out, int n) {
    int sum = 0;
    int n4 = n / 4;
    const int4* in4 = reinterpret_cast<const int4*>(in);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x) {
        int4 v = in4[i];                          // STEP 5: 128-bit 벡터화 로드
        sum += v.x + v.y + v.z + v.w;
    }
    for (int i = n4 * 4 + blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        sum += in[i];                             // STEP 4: grid-stride 꼬리 처리

    __shared__ int s[BLOCK];
    int tid = threadIdx.x;
    s[tid] = sum;
    __syncthreads();
    #pragma unroll
    for (int stride = BLOCK / 2; stride >= 64; stride >>= 1) {   // STEP 1: sequential addressing
        if (tid < stride) s[tid] += s[tid + stride];
        __syncthreads();
    }
    if (tid < 32) {                                // STEP 3: 마지막 워프는 sync-free
        int v = s[tid] + s[tid + 32];
        v = warpReduceSum(v);
        if (tid == 0) atomicAdd(out, v);
    }
}

int main(int argc, char** argv) {
    int n = (argc > 1) ? atoi(argv[1]) : (1 << 24);
    std::vector<int> h(n);
    long cpu = 0;
    for (int i = 0; i < n; ++i) { h[i] = (i % 7) - 3; cpu += h[i]; }

    DeviceBuffer<int> din(n), dout(1);
    din.copyFromHost(h.data());

    cudaDeviceProp p{}; CHECK_CUDA(cudaGetDeviceProperties(&p, 0));
    int grid = p.multiProcessorCount * 32;         // STEP 4: SM 맞춤 grid-stride
    if (grid > (n / 4 + BLOCK - 1) / BLOCK) grid = (n / 4 + BLOCK - 1) / BLOCK;

    cudaMemset(dout.data(), 0, sizeof(int));
    reduceSolution<<<grid, BLOCK>>>(din.data(), dout.data(), n);
    CHECK_CUDA(cudaDeviceSynchronize());

    GpuTimer t; t.start();
    const int iters = 200;
    for (int k = 0; k < iters; ++k) {
        cudaMemset(dout.data(), 0, sizeof(int));
        reduceSolution<<<grid, BLOCK>>>(din.data(), dout.data(), n);
    }
    double ms = t.stop() / iters;

    int got; dout.copyToHost(&got);
    double gbps = (double)n * sizeof(int) / (ms / 1e3) / 1e9;
    std::printf("solution: %.4f ms   %.1f GB/s   check=%s\n",
                ms, gbps, (got == (int)cpu) ? "OK" : "FAIL");
    return 0;
}
