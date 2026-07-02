// spmv_kernels.cuh
// 슬라이드: part5/chapter40 (SpMV, CSR) — 누적 최적화 케이스 스터디
//   L0 scalar : 행당 스레드 1개 (불규칙 행 → coalescing 나쁨)
//   L1 vector : 행당 워프 1개 + __shfl 리덕션
//   L2 +__ldg : x 벡터를 read-only 캐시로 (x는 여러 행에서 재사용 → 여기선 이득)
//   L3 ELL    : 열-major 패딩 포맷 → 스레드/행이지만 행 간 접근이 완전 coalesced
//               (리덕션·워프 낭비 없음. 규칙적 행길이일수록 이득; 편차 크면 패딩 낭비)
//
// ncu 체크포인트:
//   L0→L1 coalescing: gld sector 효율 (행/워프로 정렬 접근), l1tex 요청↓
//   L1→L2 read-only : l1tex read-only(tex) hit rate (x 재사용 → __ldg 이득)
//   L3 ELL         : gld sector 효율 (열-major → 인접 스레드가 인접 주소, 100% coalesced)
#pragma once

#include <cuda_runtime.h>

__device__ __forceinline__ float warpReduceSumF(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}

// L0: scalar CSR
__global__ void spmvScalar(const int* __restrict__ rowPtr, const int* __restrict__ colIdx,
                           const float* __restrict__ vals, const float* __restrict__ x,
                           float* __restrict__ y, int M) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M) {
        float s = 0.0f;
        for (int k = rowPtr[row]; k < rowPtr[row + 1]; ++k) s += vals[k] * x[colIdx[k]];
        y[row] = s;
    }
}

// L1/L2: vector CSR (행당 워프). UseLdg면 x를 read-only 경로로.
template <bool UseLdg>
__global__ void spmvVector(const int* __restrict__ rowPtr, const int* __restrict__ colIdx,
                           const float* __restrict__ vals, const float* __restrict__ x,
                           float* __restrict__ y, int M) {
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;
    if (warp < M) {
        int start = rowPtr[warp], end = rowPtr[warp + 1];
        float s = 0.0f;
        for (int k = start + lane; k < end; k += 32) {
            float xv;
            if constexpr (UseLdg) xv = __ldg(&x[colIdx[k]]);
            else                  xv = x[colIdx[k]];
            s += vals[k] * xv;
        }
        s = warpReduceSumF(s);
        if (lane == 0) y[warp] = s;
    }
}

// AoS vs SoA: SoA(spmvVector)는 vals[]·colIdx[] 분리, 아래는 (val,col) 인터리브 AoS.
//   실측(Orin) SoA 7.2 vs AoS 6.2 GFLOP/s(~15%): 두 필드를 함께 써 AoS도 coalesce. 페널티는 부분필드 접근 시 큼.
struct SpElem { float val; int col; };
__global__ void spmvVectorAoS(const int* __restrict__ rowPtr, const SpElem* __restrict__ elems,
                              const float* __restrict__ x, float* __restrict__ y, int M) {
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;
    if (warp < M) {
        int start = rowPtr[warp], end = rowPtr[warp + 1];
        float s = 0.0f;
        for (int k = start + lane; k < end; k += 32) {
            SpElem e = elems[k];               // AoS: val·col 함께 로드(8B, 워프 연속 → coalesced)
            s += e.val * __ldg(&x[e.col]);
        }
        s = warpReduceSumF(s);
        if (lane == 0) y[warp] = s;
    }
}

// L3: ELL 포맷 (열-major 패딩). 스레드/행이지만 j번째 비영을 ellCol[j*M+row]에서 읽어
//   같은 j에 대해 인접 스레드(row)가 인접 주소 → 완전 coalesced. 리덕션/워프 낭비 없음.
//   maxNnz = 최대 행길이(패딩 기준). 우리 입력은 K 고정이라 패딩 낭비 0.
__global__ void spmvEll(const int* __restrict__ ellCol, const float* __restrict__ ellVal,
                        const float* __restrict__ x, float* __restrict__ y,
                        int M, int maxNnz) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M) {
        float s = 0.0f;
        for (int j = 0; j < maxNnz; ++j) {
            long idx = (long)j * M + row;        // 열-major: 행 간 coalesced
            int c = ellCol[idx];
            if (c >= 0) s += ellVal[idx] * __ldg(&x[c]);   // -1 = 패딩
        }
        y[row] = s;
    }
}
