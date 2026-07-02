// matmul_kernels.cuh
// 누적 케이스 스터디 (compute-bound)
//   L0 naive     : matmulComposed<false>  (원소당 스레드, 전역 로드)
//   L1 tiled     : matmulComposed<true>   (shared memory 타일링)
//   L2 reg-tiled : sgemmRegTiled<...>      (스레드당 TM×TN, register 재사용 + ILP)
//   L3 +cp.async : sgemmRegTiledAsync<...> (더블버퍼 프리페치로 전역 로드 지연 은닉)
//   L4 WMMA      : kWmma                   (__half Tensor Core — 타입이 달라 별도 커널)
// L0/L1은 policy 플래그(Tiled)로 조합, L2/L3은 reg-tiled 계열, L4는 half 타입이라 별도.
//
// ncu 체크포인트:
//   L0→L1 재사용     : l1tex hit rate↑, dram__bytes 감소 (shared 타일링)
//   L2→L3 지연은닉   : smsp__warp_issue_stalled_long_scoreboard 감소 (LDGSTS로 로드/계산 겹침)
//   L3→L4 TensorCore : sm__pipe_tensor_op_hmma_cycles_active (HMMA 활성), roofline compute
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>   // __pipeline_memcpy_async (sm_80 미만 자동 동기 폴백)
#include <mma.h>

constexpr int TILE = 16;

// 인덱싱 선택: 행렬엔 2D(dim3)가 자연스럽다 — threadIdx.x→col 이 연속이라 C/B 접근이 coalesced,
// div/mod 불필요. 아래 matmulComposed 는 2D. 대비용 1D-flatten(div/mod) 은 matmulNaive1D.
// (1D도 idx=row*N+col 로 매핑하면 coalescing은 비슷; 차이는 div/mod 오버헤드와 가독성.)
__global__ void matmulNaive1D(const float* __restrict__ A, const float* __restrict__ B,
                              float* __restrict__ C, int M, int N, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    int row = idx / N, col = idx % N;      // div/mod (2D 인덱싱엔 없는 비용)
    float s = 0.0f;
    for (int k = 0; k < K; ++k) s += A[row * K + k] * B[k * N + col];
    C[row * N + col] = s;
}

// L0/L1: Tiled=false → naive(2D), true → shared 타일링
template <bool Tiled>
__global__ void matmulComposed(const float* __restrict__ A, const float* __restrict__ B,
                               float* __restrict__ C, int M, int N, int K) {
    int ty = threadIdx.y, tx = threadIdx.x;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    if constexpr (Tiled) {
        __shared__ float As[TILE][TILE];
        __shared__ float Bs[TILE][TILE];
        float s = 0.0f;
        for (int t = 0; t < K; t += TILE) {
            As[ty][tx] = (row < M && t + tx < K) ? A[row * K + t + tx] : 0.0f;
            Bs[ty][tx] = (t + ty < K && col < N) ? B[(t + ty) * N + col] : 0.0f;
            __syncthreads();
            #pragma unroll
            for (int k = 0; k < TILE; ++k) s += As[ty][k] * Bs[k][tx];
            __syncthreads();
        }
        if (row < M && col < N) C[row * N + col] = s;
    } else {
        if (row < M && col < N) {
            float s = 0.0f;
            for (int k = 0; k < K; ++k) s += A[row * K + k] * B[k * N + col];
            C[row * N + col] = s;
        }
    }
}

// L2 (고도화): register-tiled GEMM. 스레드당 TM×TN 출력을 레지스터에 누적 → register 재사용↑.
// blockDim = (BM/TM)*(BN/TN) 스레드. 기존 타일링(L1)의 일반화(TM=TN=1이 L1과 동일).
// __launch_bounds__: 블록당 최대 스레드수를 컴파일러에 알려 레지스터 상한/점유율을 유도.
//   (등록 레지스터가 이 스레드수에서 목표 점유율을 넘지 않게 ptxas가 조절 → 스필 위험 관리)
template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM / TM) * (BN / TN))
sgemmRegTiled(const float* __restrict__ A, const float* __restrict__ B,
                              float* __restrict__ C, int M, int N, int K) {
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];
    int blockRow = blockIdx.y * BM, blockCol = blockIdx.x * BN;
    int tRow = threadIdx.x / (BN / TN);   // 스레드의 마이크로타일 행/열
    int tCol = threadIdx.x % (BN / TN);
    const int numThreads = (BM / TM) * (BN / TN);

    // acc[TM][TN] = TM*TN 개의 독립 누산기 → 이 자체가 ILP(instruction-level parallelism).
    // 아래 내부 outer-product 는 #pragma unroll 로 완전히 펼쳐져 FMA 파이프라인을 채운다.
    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i) for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;
    float regA[TM], regB[TN];

    for (int k0 = 0; k0 < K; k0 += BK) {
        for (int i = threadIdx.x; i < BM * BK; i += numThreads) {
            int r = i / BK, c = i % BK;
            As[i] = (blockRow + r < M && k0 + c < K) ? A[(blockRow + r) * K + k0 + c] : 0.0f;
        }
        for (int i = threadIdx.x; i < BK * BN; i += numThreads) {
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
                for (int j = 0; j < TN; ++j) acc[i][j] += regA[i] * regB[j];
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

// 한 K-타일(k0)을 shared 버퍼로 async 로드. 경계는 async 대신 스칼라 0-fill (cp.async OOB 미지원).
template <int BM, int BN, int BK>
__device__ __forceinline__ void loadTileAsync(float* Asb, float* Bsb,
        const float* __restrict__ A, const float* __restrict__ B, int M, int N, int K,
        int blockRow, int blockCol, int k0, int tid, int numThreads) {
    for (int i = tid; i < BM * BK; i += numThreads) {
        int r = i / BK, c = i % BK;
        if (blockRow + r < M && k0 + c < K)
            __pipeline_memcpy_async(&Asb[i], &A[(blockRow + r) * K + k0 + c], sizeof(float));
        else Asb[i] = 0.0f;
    }
    for (int i = tid; i < BK * BN; i += numThreads) {
        int r = i / BN, c = i % BN;
        if (k0 + r < K && blockCol + c < N)
            __pipeline_memcpy_async(&Bsb[i], &B[(k0 + r) * N + blockCol + c], sizeof(float));
        else Bsb[i] = 0.0f;
    }
    __pipeline_commit();
}

// L3 (고도화): register-tiled + cp.async 더블버퍼. 현재 K-타일을 계산하는 동안 다음
//   타일을 shared 두 번째 버퍼로 LDGSTS(cp.async) 프리페치 → 전역 로드 지연을 계산과 겹침.
//   L2 와 산술은 동일(정답 불변), 파이프라인만 추가.
template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM / TM) * (BN / TN))
sgemmRegTiledAsync(const float* __restrict__ A, const float* __restrict__ B,
                                   float* __restrict__ C, int M, int N, int K) {
    __shared__ float As[2][BM * BK];
    __shared__ float Bs[2][BK * BN];
    int blockRow = blockIdx.y * BM, blockCol = blockIdx.x * BN;
    int tRow = threadIdx.x / (BN / TN);
    int tCol = threadIdx.x % (BN / TN);
    const int numThreads = (BM / TM) * (BN / TN);

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i) for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;
    float regA[TM], regB[TN];

    int nTiles = (K + BK - 1) / BK;
    loadTileAsync<BM, BN, BK>(As[0], Bs[0], A, B, M, N, K, blockRow, blockCol, 0, threadIdx.x, numThreads);
    for (int t = 0; t < nTiles; ++t) {
        int buf = t & 1;
        bool hasNext = (t + 1 < nTiles);
        if (hasNext)
            loadTileAsync<BM, BN, BK>(As[(t + 1) & 1], Bs[(t + 1) & 1], A, B, M, N, K,
                                      blockRow, blockCol, (t + 1) * BK, threadIdx.x, numThreads);
        __pipeline_wait_prior(hasNext ? 1 : 0);             // 현재 타일 도착 대기(다음은 진행중)
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            #pragma unroll
            for (int i = 0; i < TM; ++i) regA[i] = As[buf][(tRow * TM + i) * BK + kk];
            #pragma unroll
            for (int j = 0; j < TN; ++j) regB[j] = Bs[buf][kk * BN + tCol * TN + j];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                for (int j = 0; j < TN; ++j) acc[i][j] += regA[i] * regB[j];
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

// L4: WMMA Tensor Core (__half 입력, float 누적). 워프당 16×16 출력 타일.
using namespace nvcuda;
__global__ void kWmma(const half* __restrict__ A, const half* __restrict__ B,
                      float* __restrict__ C, int M, int N, int K) {
    int warpM = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    int warpN = blockIdx.y * blockDim.y + threadIdx.y;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (int k = 0; k < K; k += 16) {
        int aRow = warpM * 16, aCol = k;
        int bRow = k, bCol = warpN * 16;
        if (aRow < M && bCol < N) {
            wmma::load_matrix_sync(a_frag, A + aRow * K + aCol, K);
            wmma::load_matrix_sync(b_frag, B + bRow * N + bCol, N);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
    }
    int cRow = warpM * 16, cCol = warpN * 16;
    if (cRow < M && cCol < N)
        wmma::store_matrix_sync(C + cRow * N + cCol, c_frag, N, wmma::mem_row_major);
}
