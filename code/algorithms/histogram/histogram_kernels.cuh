// histogram_kernels.cuh
// 슬라이드: part5/chapter37 (Atomics & Histograms) — 누적 최적화 케이스 스터디
//   L0 global atomic (전역 경합)
//   L1 privatized shared (블록별 shared 히스토그램 → 병합)
//   L2 + warp-aggregated (비트 인트린직: __match_any_sync/__popc/__ffs/__activemask, SM 7.0+)
//   L3 + grid-stride
// L1~L3은 histComposed<WarpAgg,GridStride> 를 if constexpr로 조합.
//
// ncu 체크포인트:
//   L0→L1 경합     : l1tex__throughput (전역 atomic 경합↓ = shared 사유화)
//   L1→L2 점유율   : sm__warps_active (grid-stride 블록수↓)
//   L2→L3 atomic   : shared atomic 트랜잭션 수 (warp-agg; Orin은 이득 작음 = cond.)
#pragma once

#include <cuda_runtime.h>

constexpr int NBINS  = 256;
constexpr int HBLOCK = 256;

// L0: 전역 atomic (bin당 전역 경합)
__global__ void kHistGlobal(const int* __restrict__ in, int* __restrict__ hist, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        atomicAdd(&hist[in[i] & (NBINS - 1)], 1);
}

// shared 히스토그램에 1 증가. WarpAgg면 같은 bin 레인을 비트 인트린직으로 합쳐 atomic 1회.
template <bool WarpAgg>
__device__ __forceinline__ void histAdd(int* sh, int val) {
    int bin = val & (NBINS - 1);
    if constexpr (WarpAgg) {
        unsigned active = __activemask();
        unsigned peers  = __match_any_sync(active, bin);  // 같은 bin 레인 마스크
        int leader = __ffs(peers) - 1;                    // 최저 레인 = 리더
        int count  = __popc(peers);                       // 같은 bin 개수
        if ((threadIdx.x & 31) == leader) atomicAdd(&sh[bin], count);
    } else {
        atomicAdd(&sh[bin], 1);
    }
}

// L4 (고도화): sub-histograms — shared 히스토그램을 R개 복제해 atomic 경합을 분산.
template <int R>
__global__ void histSubHist(const int* __restrict__ in, int* __restrict__ hist, int n) {
    __shared__ int sh[R][NBINS];
    int tid = threadIdx.x;
    for (int r = 0; r < R; ++r)
        for (int b = tid; b < NBINS; b += blockDim.x) sh[r][b] = 0;
    __syncthreads();
    int rep = tid % R;                        // 스레드마다 다른 replica → 경합 분산
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += gridDim.x * blockDim.x)
        atomicAdd(&sh[rep][in[i] & (NBINS - 1)], 1);
    __syncthreads();
    for (int b = tid; b < NBINS; b += blockDim.x) {
        int s = 0;
        #pragma unroll
        for (int r = 0; r < R; ++r) s += sh[r][b];
        if (s) atomicAdd(&hist[b], s);
    }
}

template <bool WarpAgg, bool GridStride>
__global__ void histComposed(const int* __restrict__ in, int* __restrict__ hist, int n) {
    __shared__ int sh[NBINS];
    for (int b = threadIdx.x; b < NBINS; b += blockDim.x) sh[b] = 0;
    __syncthreads();

    if constexpr (GridStride) {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
            histAdd<WarpAgg>(sh, in[i]);
    } else {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) histAdd<WarpAgg>(sh, in[i]);
    }
    __syncthreads();

    for (int b = threadIdx.x; b < NBINS; b += blockDim.x)
        if (sh[b]) atomicAdd(&hist[b], sh[b]);
}
