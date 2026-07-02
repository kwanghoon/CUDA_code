// sort_kernels.cuh
// 슬라이드: part5/chapter36 (Sorting) — bitonic sort 누적 케이스 스터디
//   L0 global : 모든 (k,j) 스테이지를 전역 메모리 커널 런치로
//   L1 shared : j < SBLOCK 인 tail 스테이지들을 한 번의 shared 커널로 병합
//   L2 local  : k <= SBLOCK 초기 phase 전체(모든 k,j)를 shared 한 커널로 로컬 정렬
// N은 2의 거듭제곱이어야 한다. (bitonic 은 O(N·log²N) 비교-교환 = 다중 DRAM 패스)
//
// radix : 별도 알고리즘 (LSD, O(N·bits/RADIX) 패스). bitonic 의 알고리즘적 헤드룸을 보여준다.
//         메모리 roofline 대비 bitonic이 압도적으로 낮은 이유는 패스 수(≈log²N) 때문 → radix로 해결.
//
// ncu 체크포인트:
//   L0→L1 전역왕복 : dram__bytes (shared tail로 전역 접근↓), 커널 런치 수↓ (nsys로도 확인)
//   L1→L2 런치수   : 초기 log²(SBLOCK)개 전역 스테이지 → 커널 1개 (nsys 타임라인서 확연)
//   bitonic→radix  : 총 커널/패스 수 (nsys 타임라인서 O(log²N) → O(4) 로 급감)
#pragma once

#include <cuda_runtime.h>

constexpr int SBLOCK = 256;

// LSD radix sort (별도 알고리즘, 병렬)
// 4-bit digit × 8 패스. 부호 정수는 최상위 비트 XOR로 unsigned 순서화.
// 각 패스: 블록별 히스토그램(shared atomic) → 전역 오프셋 스캔 → 블록-지역 안정정렬 후 scatter.
// 블록-지역 안정정렬은 warp-multisplit(__match_any_sync/__popc)로 병렬화 — thread-serial 아님.
//   O(N) 패스라는 알고리즘적 우위를 '실제로' 살리기 위한 구현.
constexpr int R4_SIZE = 16;      // 4-bit digit
constexpr int R4_BLK  = 256;     // 블록당 원소 = blockDim (1 elem/thread)

__global__ void radixFlip(const int* __restrict__ in, int* __restrict__ out, int N, unsigned mask) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x)
        out[i] = (int)((unsigned)in[i] ^ mask);
}

// 블록별 4-bit digit 히스토그램 (shared atomic, 병렬) → blockHist[digit*numBlocks + block]
__global__ void radix4Hist(const int* __restrict__ in, int N, int shift,
                           int* __restrict__ blockHist, int numBlocks) {
    __shared__ int sh[R4_SIZE];
    int t = threadIdx.x;
    if (t < R4_SIZE) sh[t] = 0;
    __syncthreads();
    int i = blockIdx.x * R4_BLK + t;
    if (i < N) atomicAdd(&sh[((unsigned)in[i] >> shift) & (R4_SIZE - 1)], 1);
    __syncthreads();
    if (t < R4_SIZE) blockHist[t * numBlocks + blockIdx.x] = sh[t];   // bin-major
}

// 전역 exclusive 오프셋 스캔 (bin-major → 같은 digit이 블록순으로 연속 = 안정성 유지).
// total = R4_SIZE*numBlocks. 1블록 2-레벨 스캔(스레드별 청크 → 청크합 스캔 → 재분배)으로 병렬화.
__global__ void radixScanOffsets(const int* __restrict__ blockHist, int* __restrict__ offset, int total) {
    extern __shared__ int cs[];                 // blockDim 개 청크합
    int nt = blockDim.x, t = threadIdx.x;
    int per = (total + nt - 1) / nt;
    int start = t * per, end = (start + per < total) ? start + per : total;
    if (start > total) start = total;
    int sum = 0;
    for (int i = start; i < end; ++i) sum += blockHist[i];   // 내 청크 총합
    cs[t] = sum;
    __syncthreads();
    if (t == 0) { int acc = 0; for (int k = 0; k < nt; ++k) { int c = cs[k]; cs[k] = acc; acc += c; } }
    __syncthreads();
    int acc = cs[t];
    for (int i = start; i < end; ++i) { offset[i] = acc; acc += blockHist[i]; }   // 재분배
}

// 블록-지역 안정정렬(warp-multisplit) 후 전역 위치로 scatter.
__global__ void radix4Scatter(const int* __restrict__ in, int* __restrict__ out, int N, int shift,
                              const int* __restrict__ offset, int numBlocks) {
    __shared__ int warpBin[R4_BLK / 32][R4_SIZE];   // 워프별 bin 카운트
    __shared__ int binBase[R4_SIZE];                // 블록-지역 bin base
    __shared__ int sTile[R4_BLK];                   // 블록-지역 정렬 결과
    int t = threadIdx.x, lane = t & 31, wid = t >> 5;
    int gi = blockIdx.x * R4_BLK + t;
    bool active = gi < N;
    int key = active ? in[gi] : 0;
    int d = active ? (int)(((unsigned)key >> shift) & (R4_SIZE - 1)) : R4_SIZE;

    for (int k = t; k < (R4_BLK / 32) * R4_SIZE; k += R4_BLK) ((int*)warpBin)[k] = 0;
    __syncthreads();

    // intra-warp multisplit: 같은 digit 레인들 중 내 앞 개수(intraRank) + 워프 내 총개수
    int intraRank = 0;
    if (active) {
        unsigned peers = __match_any_sync(__activemask(), d);
        intraRank = __popc(peers & ((1u << lane) - 1));
        int leader = __ffs(peers) - 1;
        if (lane == leader) warpBin[wid][d] = __popc(peers);
    }
    __syncthreads();

    // 워프 간 exclusive 스캔(bin별) + bin 총합
    if (t < R4_SIZE) {
        int acc = 0;
        #pragma unroll
        for (int w = 0; w < R4_BLK / 32; ++w) { int c = warpBin[w][t]; warpBin[w][t] = acc; acc += c; }
        binBase[t] = acc;                       // 임시: bin 총개수
    }
    __syncthreads();
    if (t == 0) { int acc = 0; for (int b = 0; b < R4_SIZE; ++b) { int c = binBase[b]; binBase[b] = acc; acc += c; } }
    __syncthreads();

    if (active) sTile[binBase[d] + warpBin[wid][d] + intraRank] = key;   // 블록-지역 stable 배치
    __syncthreads();

    int cnt = (blockIdx.x * R4_BLK + R4_BLK <= N) ? R4_BLK : (N - blockIdx.x * R4_BLK);
    if (t < cnt) {
        int k2 = sTile[t];
        int d2 = ((unsigned)k2 >> shift) & (R4_SIZE - 1);
        out[offset[d2 * numBlocks + blockIdx.x] + (t - binBase[d2])] = k2;   // 전역 위치
    }
}

// 한 스테이지 (전역): i 와 i^j 를 비교-교환
__global__ void bitonicStepGlobal(int* __restrict__ data, int j, int k, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int ixj = i ^ j;
    if (ixj > i) {
        bool up = ((i & k) == 0);
        if ((up && data[i] > data[ixj]) || (!up && data[i] < data[ixj])) {
            int t = data[i]; data[i] = data[ixj]; data[ixj] = t;
        }
    }
}

// j = jHigh, jHigh/2, ..., 1 (모두 < SBLOCK) 스테이지를 shared에서 한 번에
__global__ void bitonicSharedLowJ(int* __restrict__ data, int k, int jHigh, int n) {
    __shared__ int s[SBLOCK];
    int tid  = threadIdx.x;
    int base = blockIdx.x * SBLOCK;
    s[tid] = data[base + tid];
    __syncthreads();
    for (int j = jHigh; j > 0; j >>= 1) {
        int i   = base + tid;    // 전역 인덱스(정렬 방향 결정)
        int lxj = tid ^ j;       // 블록 내 파트너 (j < SBLOCK 이라 같은 블록)
        if (lxj > tid) {
            bool up = ((i & k) == 0);
            if ((up && s[tid] > s[lxj]) || (!up && s[tid] < s[lxj])) {
                int t = s[tid]; s[tid] = s[lxj]; s[lxj] = t;
            }
        }
        __syncthreads();
    }
    data[base + tid] = s[tid];
}

// L2: k <= SBLOCK 초기 phase 전체를 shared에서 로컬 정렬. 이 구간은 모든 (k,j)에서
//   j = k/2 < SBLOCK 이므로 파트너가 항상 같은 블록 → 전역 왕복 없이 한 커널로 끝낸다.
//   방향 비트는 전역 인덱스(base+tid)&k 로 결정해 전역 bitonic 네트워크와 동일.
__global__ void bitonicLocalSort(int* __restrict__ data, int n) {
    __shared__ int s[SBLOCK];
    int tid  = threadIdx.x;
    int base = blockIdx.x * SBLOCK;
    s[tid] = data[base + tid];
    __syncthreads();
    for (int k = 2; k <= SBLOCK; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            int i   = base + tid;
            int lxj = tid ^ j;
            if (lxj > tid) {
                bool up = ((i & k) == 0);
                if ((up && s[tid] > s[lxj]) || (!up && s[tid] < s[lxj])) {
                    int t = s[tid]; s[tid] = s[lxj]; s[lxj] = t;
                }
            }
            __syncthreads();
        }
    }
    data[base + tid] = s[tid];
}
