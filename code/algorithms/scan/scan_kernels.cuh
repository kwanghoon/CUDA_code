// scan_kernels.cuh
// 슬라이드: part5/chapter38 (Scan) — 누적 최적화 케이스 스터디
//   L0 sequential      (세그먼트당 스레드 1개, 직렬)
//   L1 Hillis-Steele   (shared memory, ping-pong 더블버퍼)   scanComposed<false>
//   L2 warp-shuffle    (register __shfl, smem/sync 최소)      scanComposed<true>
//   L3 +cp.async       (grid-stride 프리페치 더블버퍼)         kAsyncPrefetch
//   L4 int4+coarsening (128-bit 로드, 스레드당 4원소 레지스터 스캔)  kScanVec4
// L1/L2는 하나의 템플릿을 policy 플래그(WarpShuffle)로 조합한다.
//
// ncu 체크포인트:
//   L0→L1 병렬화   : sm__warps_active (직렬 baseline 대비 점유율↑)
//   L1→L2 smem/sync: l1tex__data_bank_conflicts_pipe_lsu_mem_shared, __syncthreads 감소
//   L2→L3 전송오버랩: cp.async(LDGSTS) 활용, smsp__warp_issue_stalled_long_scoreboard 감소
//   L3→L4 벡터화   : gld/gst 트랜잭션 수(int4=128-bit 1회), __syncthreads 절반(64스레드/2워프)
#pragma once

#include <cuda_runtime.h>
#include <cuda_pipeline.h>   // __pipeline_memcpy_async (sm_80 미만 자동 동기 폴백)

constexpr int SEG            = 256;
constexpr int NUM_WARPS      = SEG / 32;
constexpr int ASYNC_GRID_CAP = 2048;

// cuda-gdb: break scanComposed; print buf[pin][tid] / print/x __activemask()
//           race 의심: compute-sanitizer --tool racecheck ./scan_bench

__device__ __forceinline__ int warpInclusiveScan(int val) {
    int lane = threadIdx.x & 31;
    #pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
        int n = __shfl_up_sync(0xffffffffu, val, off);
        if (lane >= off) val += n;
    }
    return val;
}

// 블록 inclusive scan: 워프 스캔 → 워프합 스캔 → 오프셋 합산
__device__ __forceinline__ int blockInclusiveScan(int val, int* warpSums) {
    int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    val = warpInclusiveScan(val);
    if (lane == 31) warpSums[wid] = val;
    __syncthreads();
    if (wid == 0) {
        int ws = (tid < NUM_WARPS) ? warpSums[tid] : 0;
        ws = warpInclusiveScan(ws);
        if (tid < NUM_WARPS) warpSums[tid] = ws;
    }
    __syncthreads();
    if (wid > 0) val += warpSums[wid - 1];
    return val;
}

// L0: sequential — 세그먼트당 스레드 1개(블록에 패킹). 세그먼트 내부 병렬성 없음(baseline).
__global__ void kSequential(const int* __restrict__ in, int* __restrict__ out, int numSeg) {
    int seg = blockIdx.x * blockDim.x + threadIdx.x;
    if (seg >= numSeg) return;
    int base = seg * SEG, acc = 0;
    for (int i = 0; i < SEG; ++i) { acc += in[base + i]; out[base + i] = acc; }
}

// L1/L2: 블록당 세그먼트 1개. WarpShuffle=false → Hillis-Steele(shared), true → warp-shuffle.
template <bool WarpShuffle>
__global__ void scanComposed(const int* __restrict__ in, int* __restrict__ out) {
    int tid = threadIdx.x, gid = blockIdx.x * SEG + tid;
    int val = in[gid];
    if constexpr (WarpShuffle) {
        __shared__ int warpSums[NUM_WARPS];
        out[gid] = blockInclusiveScan(val, warpSums);
    } else {
        // Hillis-Steele: predication(분기 아님)으로 divergence 없음
        __shared__ int buf[2][SEG];
        int pin = 0, pout = 1;
        buf[pin][tid] = val;
        __syncthreads();
        for (int off = 1; off < SEG; off <<= 1) {
            buf[pout][tid] = (tid >= off) ? buf[pin][tid] + buf[pin][tid - off]
                                          : buf[pin][tid];
            __syncthreads();
            pin ^= 1; pout ^= 1;
        }
        out[gid] = buf[pin][tid];
    }
}

// L3의 로드 헬퍼 (#if 없이 자동 폴백)
__device__ __forceinline__ void loadSegToShared(int* dstSmem, const int* srcGlobal) {
    __pipeline_memcpy_async(&dstSmem[threadIdx.x], &srcGlobal[threadIdx.x], sizeof(int));
    __pipeline_commit();
}
__device__ __forceinline__ void waitSeg(bool prefetchIssued) {
    __pipeline_wait_prior(prefetchIssued ? 1 : 0);
    __syncthreads();
}

// L3: cp.async 프리페치 더블버퍼 + grid-stride (producer=async load, consumer=scan)
__global__ void kAsyncPrefetch(const int* __restrict__ in, int* __restrict__ out, int numSeg) {
    __shared__ int buf[2][SEG];
    __shared__ int warpSums[NUM_WARPS];
    int tid = threadIdx.x;
    int seg = blockIdx.x;
    if (seg >= numSeg) return;

    int cur = 0;
    loadSegToShared(buf[cur], in + seg * SEG);
    while (seg < numSeg) {
        int nextSeg = seg + gridDim.x;
        bool prefetch = (nextSeg < numSeg);
        if (prefetch) loadSegToShared(buf[cur ^ 1], in + nextSeg * SEG);
        waitSeg(prefetch);
        out[seg * SEG + tid] = blockInclusiveScan(buf[cur][tid], warpSums);
        __syncthreads();
        cur ^= 1;
        seg = nextSeg;
    }
}

// L4: int4 벡터 로드 + thread-coarsening. 블록당 SEG/4 스레드, 각 스레드가 4원소를
//   레지스터에서 직렬 스캔 → 스레드 합만 블록 스캔 → 오프셋 재분배. 128-bit 로드/스토어
//   1회, 워프 수 절반(64스레드=2워프)이라 __syncthreads·bank 트래픽이 준다.
//   coarsening 블록 스캔은 실제 워프수(2)에 맞춘 전용 조합(blockInclusiveScan 재사용 불가:
//   그건 NUM_WARPS=8 가정). 세그먼트당 inclusive scan 결과는 L0..L3과 동일.
constexpr int VEC        = 4;
constexpr int VEC_THREADS = SEG / VEC;          // 64
constexpr int VEC_WARPS   = VEC_THREADS / 32;   // 2
__global__ void kScanVec4(const int* __restrict__ in, int* __restrict__ out) {
    __shared__ int warpSums[VEC_WARPS];
    int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    int base = blockIdx.x * SEG;

    int4 v = reinterpret_cast<const int4*>(in + base)[tid];  // 128-bit coalesced 로드
    v.y += v.x; v.z += v.y; v.w += v.z;                      // 스레드 내 4원소 inclusive
    int threadTotal = v.w;

    // 스레드 합의 블록 inclusive scan (워프 스캔 → 워프합 스캔 → 결합)
    int wscan = warpInclusiveScan(threadTotal);
    if (lane == 31) warpSums[wid] = wscan;
    __syncthreads();
    if (wid == 0) {
        int ws = (tid < VEC_WARPS) ? warpSums[tid] : 0;
        ws = warpInclusiveScan(ws);
        if (tid < VEC_WARPS) warpSums[tid] = ws;
    }
    __syncthreads();
    int inclusive = wscan + (wid > 0 ? warpSums[wid - 1] : 0);
    int offset = inclusive - threadTotal;                    // 이 스레드의 exclusive prefix
    v.x += offset; v.y += offset; v.z += offset; v.w += offset;
    reinterpret_cast<int4*>(out + base)[tid] = v;            // 128-bit coalesced 스토어
}
