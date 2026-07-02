// fft_kernels.cuh
// 슬라이드: part5/chapter41 (FFT) — 누적 케이스 스터디 (배치 1D FFT, 블록당 1 변환)
//   L0 naive DFT      : O(N^2), 스레드당 출력 1개
//   L1 radix-2 FFT    : Cooley-Tukey, shared memory, O(N log N)
//   L2 +fast twiddle  : cosf+sinf 2회 → __sincosf 1회 (fast-math SFU)
// L1/L2는 fftShared<FastTwiddle> 를 policy 플래그로 조합. complex = float2.
// N은 2의 거듭제곱, 블록 하나가 한 변환을 담당.
//
// ncu 체크포인트:
//   L0→L1 복잡도 : inst_executed 급감 (O(N²)→O(N log N)), compute·dram 모두 감소
//   L1→L2 SFU    : smsp__sass_thread_inst_executed_op_*_pred (초월함수 명령↓), MUFU 사용률
#pragma once

#include <cuda_runtime.h>

#ifndef FFT_PI
#define FFT_PI 3.14159265358979323846f
#endif

__device__ __forceinline__ float2 cadd(float2 a, float2 b) { return make_float2(a.x + b.x, a.y + b.y); }
__device__ __forceinline__ float2 csub(float2 a, float2 b) { return make_float2(a.x - b.x, a.y - b.y); }
__device__ __forceinline__ float2 cmul(float2 a, float2 b) {
    return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// L0: naive DFT. 블록당 N 스레드, 각 스레드가 X[k] = sum_n x[n] e^{-2pi i kn/N}
__global__ void dftNaive(const float2* __restrict__ in, float2* __restrict__ out, int N) {
    int k = threadIdx.x;
    int base = blockIdx.x * N;
    if (k >= N) return;
    float2 s = make_float2(0.0f, 0.0f);
    for (int n = 0; n < N; ++n) {
        float ang = -2.0f * FFT_PI * k * n / N;
        float2 w = make_float2(cosf(ang), sinf(ang));
        s = cadd(s, cmul(in[base + n], w));
    }
    out[base + k] = s;
}

__device__ __forceinline__ int bitrev(int x, int bits) {
    int r = 0;
    for (int i = 0; i < bits; ++i) { r = (r << 1) | (x & 1); x >>= 1; }
    return r;
}

// 트위들 w = e^{-i·ang}. FastTwiddle면 __sincosf(SFU 1회), 아니면 cosf+sinf(2회).
template <bool FastTwiddle>
__device__ __forceinline__ float2 twiddle(float ang) {
    if constexpr (FastTwiddle) {
        float sn, cs;
        __sincosf(ang, &sn, &cs);          // 단일 MUFU 명령으로 sin·cos 동시
        return make_float2(cs, sn);
    } else {
        return make_float2(cosf(ang), sinf(ang));
    }
}

// L1/L2: radix-2 Cooley-Tukey FFT. 블록당 N/2 스레드, shared에서 in-place.
template <bool FastTwiddle>
__global__ void fftShared(const float2* __restrict__ in, float2* __restrict__ out, int N, int bits) {
    extern __shared__ float2 s[];
    int t = threadIdx.x;              // 0 .. N/2-1
    int base = blockIdx.x * N;

    // bit-reversal 순서로 로드 (스레드당 2개)
    s[bitrev(2 * t,     bits)] = in[base + 2 * t];
    s[bitrev(2 * t + 1, bits)] = in[base + 2 * t + 1];
    __syncthreads();

    for (int len = 2; len <= N; len <<= 1) {
        int half = len >> 1;
        int group = t / half;
        int pos   = t % half;
        int i = group * len + pos;
        int j = i + half;
        float ang = -2.0f * FFT_PI * pos / len;
        float2 w = twiddle<FastTwiddle>(ang);
        float2 a = s[i], b = cmul(s[j], w);
        s[i] = cadd(a, b);
        s[j] = csub(a, b);
        __syncthreads();
    }

    out[base + 2 * t]     = s[2 * t];
    out[base + 2 * t + 1] = s[2 * t + 1];
}

// AoS vs SoA: fftShared는 복소수 = float2(실/허 인터리브 = AoS-of-complex). 아래는 SoA —
//   real[]·imag[] 분리 배열, shared도 sre/sim 분리. 버터플라이가 실·허를 '함께' 쓰므로
//   AoS(float2 8B 벡터 로드)·SoA(두 coalesced 스트림) 둘 다 효율적 → 비슷(작은 N서 노이즈 큼).
//   SoA가 확실히 유리한 건 한 성분만 접근할 때(예: 크기 |z|² 만). 여기선 부분접근 아님.
__global__ void fftSharedSoA(const float* __restrict__ inRe, const float* __restrict__ inIm,
                             float* __restrict__ outRe, float* __restrict__ outIm, int N, int bits) {
    extern __shared__ float smem[];        // [0..N)=실수, [N..2N)=허수
    float* sre = smem;
    float* sim = smem + N;
    int t = threadIdx.x, base = blockIdx.x * N;
    int r0 = bitrev(2 * t, bits), r1 = bitrev(2 * t + 1, bits);
    sre[r0] = inRe[base + 2 * t];     sim[r0] = inIm[base + 2 * t];
    sre[r1] = inRe[base + 2 * t + 1]; sim[r1] = inIm[base + 2 * t + 1];
    __syncthreads();

    for (int len = 2; len <= N; len <<= 1) {
        int half = len >> 1, group = t / half, pos = t % half;
        int i = group * len + pos, j = i + half;
        float ang = -2.0f * FFT_PI * pos / len;
        float wr = cosf(ang), wi = sinf(ang);
        float are = sre[i], aim = sim[i], bre0 = sre[j], bim0 = sim[j];
        float bre = bre0 * wr - bim0 * wi, bim = bre0 * wi + bim0 * wr;   // b = s[j]*w
        sre[i] = are + bre; sim[i] = aim + bim;
        sre[j] = are - bre; sim[j] = aim - bim;
        __syncthreads();
    }
    outRe[base + 2 * t]     = sre[2 * t];     outIm[base + 2 * t]     = sim[2 * t];
    outRe[base + 2 * t + 1] = sre[2 * t + 1]; outIm[base + 2 * t + 1] = sim[2 * t + 1];
}
