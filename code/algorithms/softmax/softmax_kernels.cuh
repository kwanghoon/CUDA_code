// softmax_kernels.cuh
// 슬라이드: part5/chapter44 (ML Layer Patterns) — row-wise stable softmax 누적 케이스 스터디
//   L1 <F,F,F> : block/row 병렬 (행당 스레드 → 블록당 행)
//   L2 <T,T,F> : + 행 shared 캐시(global 읽기 3→1) + warp-shuffle 리덕션
//   L3 <T,T,T> : + fast reciprocal (행별 나눗셈 col회 → __fdividef 역수 1회 후 곱)
// 블록당 행 1개. numerically stable (max 빼기). (L0 serial 은 별도 baseline)
#pragma once

#include <cuda_runtime.h>

constexpr int SM_BLK = 256;

__device__ __forceinline__ float warpMax(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, o));
    return v;
}
__device__ __forceinline__ float warpSum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}

template <bool Warp>
__device__ __forceinline__ float blockMax(float v, float* sh) {
    int t = threadIdx.x;
    if constexpr (Warp) {
        v = warpMax(v);
        int lane = t & 31, wid = t >> 5;
        if (lane == 0) sh[wid] = v;
        __syncthreads();
        if (wid == 0) { v = (t < blockDim.x / 32) ? sh[t] : -1e30f; v = warpMax(v); if (t == 0) sh[0] = v; }
        __syncthreads();
        float r = sh[0]; __syncthreads(); return r;
    } else {
        sh[t] = v; __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (t < s) sh[t] = fmaxf(sh[t], sh[t + s]); __syncthreads(); }
        float r = sh[0]; __syncthreads(); return r;
    }
}
template <bool Warp>
__device__ __forceinline__ float blockSum(float v, float* sh) {
    int t = threadIdx.x;
    if constexpr (Warp) {
        v = warpSum(v);
        int lane = t & 31, wid = t >> 5;
        if (lane == 0) sh[wid] = v;
        __syncthreads();
        if (wid == 0) { v = (t < blockDim.x / 32) ? sh[t] : 0.0f; v = warpSum(v); if (t == 0) sh[0] = v; }
        __syncthreads();
        float r = sh[0]; __syncthreads(); return r;
    } else {
        sh[t] = v; __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (t < s) sh[t] += sh[t + s]; __syncthreads(); }
        float r = sh[0]; __syncthreads(); return r;
    }
}

// L0: 행당 스레드 1개가 직렬 처리 (행 내부 병렬 없음 + uncoalesced 접근 → 느린 baseline).
__global__ void softmaxSerial(const float* __restrict__ in, float* __restrict__ out,
                              int rows, int cols) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const float* x = in + (size_t)r * cols;
    float* y = out + (size_t)r * cols;
    float m = -1e30f; for (int c = 0; c < cols; ++c) m = fmaxf(m, x[c]);
    float l = 0.0f;   for (int c = 0; c < cols; ++c) l += __expf(x[c] - m);
    for (int c = 0; c < cols; ++c) y[c] = __expf(x[c] - m) / l;
}

// L1~L3: 블록당 행 1개 (병렬 리덕션). CacheRow=행 shared 캐시, Warp=warp-shuffle 리덕션,
//   FastDiv=행별 나눗셈을 __fdividef 역수 1회로 대체 후 곱 (div는 mul보다 훨씬 비쌈).
template <bool CacheRow, bool Warp, bool FastDiv>
__global__ void softmaxComposed(const float* __restrict__ in, float* __restrict__ out,
                                int rows, int cols) {
    extern __shared__ float srow[];      // CacheRow=true 시 [cols]
    __shared__ float red[SM_BLK];
    int row = blockIdx.x, tid = threadIdx.x;
    const float* x = in + (size_t)row * cols;
    float* y = out + (size_t)row * cols;

    if constexpr (CacheRow) {
        for (int c = tid; c < cols; c += blockDim.x) srow[c] = x[c];
        __syncthreads();
    }

    float m = -1e30f;
    for (int c = tid; c < cols; c += blockDim.x) {
        float xv; if constexpr (CacheRow) xv = srow[c]; else xv = x[c];
        m = fmaxf(m, xv);
    }
    m = blockMax<Warp>(m, red);

    float l = 0.0f;
    for (int c = tid; c < cols; c += blockDim.x) {
        float xv; if constexpr (CacheRow) xv = srow[c]; else xv = x[c];
        l += __expf(xv - m);
    }
    l = blockSum<Warp>(l, red);

    float invL = FastDiv ? __fdividef(1.0f, l) : 0.0f;   // 역수 1회 (FastDiv)
    for (int c = tid; c < cols; c += blockDim.x) {
        float xv; if constexpr (CacheRow) xv = srow[c]; else xv = x[c];
        if constexpr (FastDiv) y[c] = __expf(xv - m) * invL;   // 곱 (div 제거)
        else                   y[c] = __expf(xv - m) / l;
    }
}
