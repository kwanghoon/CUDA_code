// attention_kernels.cuh
// 슬라이드: part10/chapter77 (Attention & FlashAttention) — 누적 케이스 스터디
//   L0 naive : 블록당 쿼리 1행, scores[N]를 shared에 전개 후 softmax → PV
//   L1 flash : online softmax — scores 전개 없이 K/V 타일을 스트리밍하며 running max/sum/acc
// Q,K,V,O : [N, HEAD_DIM] (단일 헤드). blockDim = HEAD_DIM (스레드 t = 출력 차원).
#pragma once

#include <cuda_runtime.h>

constexpr int HEAD_DIM = 64;
constexpr int FLASH_TN = 64;   // flash K/V 타일 크기

__device__ __forceinline__ float blockReduceMax(float v, float* red) {
    int t = threadIdx.x;
    red[t] = v; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (t < s) red[t] = fmaxf(red[t], red[t + s]); __syncthreads(); }
    float r = red[0]; __syncthreads(); return r;
}
__device__ __forceinline__ float blockReduceSum(float v, float* red) {
    int t = threadIdx.x;
    red[t] = v; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (t < s) red[t] += red[t + s]; __syncthreads(); }
    float r = red[0]; __syncthreads(); return r;
}

// L0: scores[N]를 shared에 전개 (O(N) shared, 긴 시퀀스에서 부담)
__global__ void attnNaive(const float* __restrict__ Q, const float* __restrict__ K,
                          const float* __restrict__ V, float* __restrict__ O, int N) {
    extern __shared__ float sh[];
    float* scores = sh;                       // [N]
    float* q      = sh + N;                   // [HEAD_DIM]
    float* red    = sh + N + HEAD_DIM;         // [blockDim]
    int i = blockIdx.x, t = threadIdx.x;
    q[t] = Q[i * HEAD_DIM + t];
    __syncthreads();
    float scale = rsqrtf((float)HEAD_DIM);

    for (int j = t; j < N; j += blockDim.x) {
        float s = 0.0f;
        for (int k = 0; k < HEAD_DIM; ++k) s += q[k] * K[j * HEAD_DIM + k];
        scores[j] = s * scale;
    }
    __syncthreads();

    float m = -1e30f;
    for (int j = t; j < N; j += blockDim.x) m = fmaxf(m, scores[j]);
    m = blockReduceMax(m, red);
    float l = 0.0f;
    for (int j = t; j < N; j += blockDim.x) { float e = __expf(scores[j] - m); scores[j] = e; l += e; }
    l = blockReduceSum(l, red);
    __syncthreads();

    float acc = 0.0f;                          // t = 출력 차원
    for (int j = 0; j < N; ++j) acc += scores[j] * V[j * HEAD_DIM + t];
    O[i * HEAD_DIM + t] = acc / l;
}

// per-tensor 로드 전략: K/V는 read-only 캐시(__ldg), Q는 streaming(__ldcs).
template <bool CacheKV> __device__ __forceinline__ float loadKV(const float* p) {
    if constexpr (CacheKV) return __ldg(p);      // 재사용 → read-only/L1 캐시
    else                   return *p;
}
template <bool CacheKV> __device__ __forceinline__ float loadQ(const float* p) {
    if constexpr (CacheKV) return __ldcs(p);     // 일회성 → streaming(캐시 오염 방지)
    else                   return *p;
}

// L1/L2: FlashAttention — online softmax. CacheKV=true면 K/V read-only + Q streaming.
template <bool CacheKV>
__global__ void attnFlash(const float* __restrict__ Q, const float* __restrict__ K,
                          const float* __restrict__ V, float* __restrict__ O, int N) {
    __shared__ float q[HEAD_DIM];
    __shared__ float ss[FLASH_TN];
    __shared__ float red[HEAD_DIM];
    int i = blockIdx.x, t = threadIdx.x;      // t = 출력 차원
    q[t] = loadQ<CacheKV>(&Q[i * HEAD_DIM + t]);
    __syncthreads();
    float scale = rsqrtf((float)HEAD_DIM);

    float m = -1e30f, l = 0.0f, acc = 0.0f;    // running max/sum/output(dim t)
    for (int j0 = 0; j0 < N; j0 += FLASH_TN) {
        int tn = (N - j0 < FLASH_TN) ? (N - j0) : FLASH_TN;
        for (int jj = 0; jj < tn; ++jj) {
            float part = q[t] * loadKV<CacheKV>(&K[(j0 + jj) * HEAD_DIM + t]);
            float dot = blockReduceSum(part, red);
            if (t == 0) ss[jj] = dot * scale;
        }
        __syncthreads();

        float tileMax = -1e30f;
        for (int jj = 0; jj < tn; ++jj) tileMax = fmaxf(tileMax, ss[jj]);
        float mNew = fmaxf(m, tileMax);
        float corr = __expf(m - mNew);
        acc *= corr; l *= corr;
        for (int jj = 0; jj < tn; ++jj) {
            float p = __expf(ss[jj] - mNew);
            l   += p;
            acc += p * loadKV<CacheKV>(&V[(j0 + jj) * HEAD_DIM + t]);
        }
        m = mNew;
        __syncthreads();
    }
    O[i * HEAD_DIM + t] = acc / l;
}
