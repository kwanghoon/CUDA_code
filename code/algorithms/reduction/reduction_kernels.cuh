// reduction_kernels.cuh
// 누적 최적화 케이스 스터디 (ch20 스타일)
// 하나의 템플릿 커널을 policy 플래그로 조합(if constexpr). 각 레벨 = 이전 + 플래그 1개.
//   L0 <F,F,F,F> divergent 트리(느린 baseline)
//   L1 <T,F,F,F> + sequential-addressing (divergence/bank-conflict 제거)
//   L2 <T,T,F,F> + warp-tail (마지막 워프 sync-free)
//   L3 <T,T,T,F> + grid-stride (블록 수 SM 맞춤)
//   L4 <T,T,T,T> + int4 벡터화 로드   ← 여기까지가 누적 사다리(각 레벨 = 이전 + 기법1개, 단조)
// 아래 A1/A2/A3 는 누적이 아니라 L4 위의 "대체 메커니즘" 데모 (택1, 서로 못 쌓임):
//   A1 single-pass  __threadfence() + 원자 카운터로 마지막 블록이 결합 (2차 런치 제거)
//   A2 coop-groups  grid.sync() 전-그리드 배리어 단일패스
//   A3 HW warp-reduce __reduce_add_sync (Ampere+ 단일 명령)
//
// ncu 체크포인트 (ncu --set full 또는 개별 --metrics):
//   L0→L1 divergence: smsp__thread_inst_executed_per_inst_executed (1.0에 가까울수록 개선)
//   L1→L2 bank/sync : l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld, __syncthreads 감소
//   L2→L3 occupancy : sm__warps_active.avg.pct_of_peak_sustained_active (블록수↓·점유율)
//   L3→L4 대역폭    : dram__throughput.avg.pct_of_peak_sustained_elapsed (int4로 상승)
#pragma once

#include <cuda_runtime.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

constexpr int RBLOCK = 256;

__device__ __forceinline__ int warpReduceSum(int v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}

// HW 워프 리듀스: Ampere(sm_80)+는 __reduce_add_sync 단일 명령. 이전 아키텍처는 셔플 폴백.
// (__reduce_*_sync는 sm_80 미만에서 컴파일 불가 → 여기선 __CUDA_ARCH__ 가드가 정당한 경우)
__device__ __forceinline__ int warpReduceSumHW(int v) {
#if __CUDA_ARCH__ >= 800
    return (int)__reduce_add_sync(0xffffffffu, (unsigned)v);
#else
    return warpReduceSum(v);
#endif
}

template <bool Sequential, bool WarpTail, bool GridStride, bool Vec4>
__global__ void reduceComposed(const int* __restrict__ in, int* __restrict__ out, int n) {
    int sum = 0;

    if constexpr (GridStride) {
        if constexpr (Vec4) {
            int n4 = n / 4;
            const int4* in4 = reinterpret_cast<const int4*>(in);
            for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x) {
                int4 v = in4[i];
                sum += v.x + v.y + v.z + v.w;
            }
            for (int i = n4 * 4 + blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
                sum += in[i];
        } else {
            for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
                sum += in[i];
        }
    } else {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        sum = (i < n) ? in[i] : 0;
    }

    __shared__ int s[RBLOCK];
    int tid = threadIdx.x;
    s[tid] = sum;
    __syncthreads();

    if constexpr (!Sequential) {
        // divergent interleaved-addressing: if(tid % (2*st)==0) — warp divergence + bank conflict
        for (int st = 1; st < blockDim.x; st <<= 1) {
            if (tid % (2 * st) == 0) s[tid] += s[tid + st];
            __syncthreads();
        }
        if (tid == 0) atomicAdd(out, s[0]);
    } else if constexpr (!WarpTail) {
        // sequential addressing: if(tid<st) — 연속 스레드만 활성(divergence-free)
        for (int st = blockDim.x / 2; st > 0; st >>= 1) {
            if (tid < st) s[tid] += s[tid + st];
            __syncthreads();
        }
        if (tid == 0) atomicAdd(out, s[0]);
    } else {
        // + 마지막 워프 sync-free (__shfl)
        #pragma unroll
        for (int st = RBLOCK / 2; st >= 64; st >>= 1) {
            if (tid < st) s[tid] += s[tid + st];
            __syncthreads();
        }
        if (tid < 32) {
            int v = s[tid] + s[tid + 32];
            v = warpReduceSum(v);
            if (tid == 0) atomicAdd(out, v);
        }
    }
}

// L5 (고도화): single-pass reduction — __threadfence()로 부분합 가시화 후 원자 카운터로
//   마지막 블록이 결합 (2차 런치 제거). fence 없으면 약한 메모리 순서로 부분합 못 봄.
__device__ unsigned g_retCount = 0;     // 은퇴한 블록 수 (atomicInc가 gridDim-1에서 0으로 랩)
__device__ int      g_partials[1024];

__global__ void reduceSinglePass(const int* __restrict__ in, int* __restrict__ out, int n) {
    // L4까지 누적: int4 벡터화 grid-stride 로드
    int sum = 0;
    int n4 = n / 4;
    const int4* in4 = reinterpret_cast<const int4*>(in);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x) {
        int4 v4 = in4[i];
        sum += v4.x + v4.y + v4.z + v4.w;
    }
    for (int i = n4 * 4 + blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        sum += in[i];

    // warp-shuffle + 워프합 결합
    __shared__ int s[RBLOCK / 32];
    int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    sum = warpReduceSum(sum);
    if (lane == 0) s[wid] = sum;
    __syncthreads();
    if (wid != 0) return;
    int v = (tid < RBLOCK / 32) ? s[tid] : 0;
    v = warpReduceSum(v);
    if (tid != 0) return;

    g_partials[blockIdx.x] = v;
    __threadfence();                                      // 부분합 쓰기를 전 그리드에 가시화
    unsigned done = atomicInc(&g_retCount, gridDim.x - 1);// gridDim-1 도달 시 0으로 자동 리셋
    if (done == gridDim.x - 1) {
        int total = 0;
        for (int b = 0; b < gridDim.x; ++b) total += g_partials[b];  // fence 덕에 안전
        *out = total;
    }
}

// int4 grid-stride 로드 + 블록 리덕션 → 블록 부분합(모든 스레드가 동일 값 보유).
__device__ __forceinline__ int blockReduceInt4(const int* __restrict__ in, int n) {
    int sum = 0, n4 = n / 4;
    const int4* in4 = reinterpret_cast<const int4*>(in);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x) {
        int4 v4 = in4[i];
        sum += v4.x + v4.y + v4.z + v4.w;
    }
    for (int i = n4 * 4 + blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        sum += in[i];
    __shared__ int s[RBLOCK / 32];
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    sum = warpReduceSum(sum);
    if (lane == 0) s[wid] = sum;
    __syncthreads();
    int v = (threadIdx.x < RBLOCK / 32) ? s[threadIdx.x] : 0;
    v = warpReduceSum(v);
    __shared__ int bsum;
    if (threadIdx.x == 0) bsum = v;
    __syncthreads();
    return bsum;                          // 전 스레드 브로드캐스트
}

// L6 (고도화): coop-groups grid.sync() single-pass. grid.sync() 전-그리드 배리어로 phase 분리.
//   협력 런치 필요 — 모든 블록 동시 상주해야 해 그리드를 occupancy로 제한.
__global__ void reduceCoopGroups(const int* __restrict__ in, int* __restrict__ out, int n) {
    cg::grid_group grid = cg::this_grid();
    int part = blockReduceInt4(in, n);                    // phase 1: 블록별 부분합
    if (threadIdx.x == 0) g_partials[blockIdx.x] = part;
    grid.sync();                                          // 전-그리드 배리어 (fence+sync 내포)
    if (blockIdx.x == 0) {                                // phase 2: 블록0가 부분합 결합
        int acc = 0;
        for (int b = threadIdx.x; b < gridDim.x; b += blockDim.x) acc += g_partials[b];
        __shared__ int s[RBLOCK / 32];
        int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
        acc = warpReduceSum(acc);
        if (lane == 0) s[wid] = acc;
        __syncthreads();
        if (wid == 0) {
            int v = (threadIdx.x < RBLOCK / 32) ? s[threadIdx.x] : 0;
            v = warpReduceSum(v);
            if (threadIdx.x == 0) *out = v;
        }
    }
}

// L7 (고도화): L4(int4) + HW 워프 리듀스(__reduce_add_sync). 워프 내 5회 셔플을 단일
//   명령으로 대체. Orin에선 셔플도 빨라 이득이 작을 수 있음(cond.) — 측정이 판단.
__global__ void reduceHWWarp(const int* __restrict__ in, int* __restrict__ out, int n) {
    int sum = 0, n4 = n / 4;
    const int4* in4 = reinterpret_cast<const int4*>(in);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x) {
        int4 v4 = in4[i];
        sum += v4.x + v4.y + v4.z + v4.w;
    }
    for (int i = n4 * 4 + blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        sum += in[i];
    __shared__ int s[RBLOCK / 32];
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    sum = warpReduceSumHW(sum);                    // HW 워프 리듀스 (Ampere+ 단일 명령)
    if (lane == 0) s[wid] = sum;
    __syncthreads();
    if (wid == 0) {
        int v = (threadIdx.x < RBLOCK / 32) ? s[threadIdx.x] : 0;
        v = warpReduceSumHW(v);
        if (threadIdx.x == 0) atomicAdd(out, v);
    }
}
