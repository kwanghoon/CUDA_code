// convolution_kernels.cuh
// 슬라이드: part5/chapter35 (Convolution) — 누적 최적화 케이스 스터디
//   L0 naive        (필터를 전역 메모리에서)
//   L1 + constant   (필터를 __constant__ 브로드캐스트 캐시로)
//   L2 + shared tile (halo 포함 입력 타일 재사용)
// convComposed<UseConstant, UseShared> 를 if constexpr로 조합. 5x5 필터 고정.
//
// ncu 체크포인트:
//   L0→L1 상수캐시 : constant/uniform cache hit (필터 브로드캐스트)
//   L1→L2 재사용   : l1tex__t_sector_hit_rate (halo 타일 재사용), dram 읽기↓
#pragma once

#include <cuda_runtime.h>

constexpr int RADIUS = 2;
constexpr int FDIM   = 2 * RADIUS + 1;   // 5
constexpr int FSIZE  = FDIM * FDIM;      // 25
constexpr int CTILE  = 16;

__constant__ float cFilter[FSIZE];

__device__ __forceinline__ int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

template <bool UseConstant, bool UseShared>
__global__ void convComposed(const float* __restrict__ in, const float* __restrict__ filter,
                             float* __restrict__ out, int W, int H) {
    int tx = threadIdx.x, ty = threadIdx.y;

    if constexpr (UseShared) {
        __shared__ float tile[CTILE + 2 * RADIUS][CTILE + 2 * RADIUS];
        int x0 = blockIdx.x * CTILE, y0 = blockIdx.y * CTILE;
        // 타일(중심+halo) 로드 — (CTILE+2R)² > CTILE² 이므로 스레드가 여러 셀 로드
        for (int j = ty; j < CTILE + 2 * RADIUS; j += CTILE)
            for (int i = tx; i < CTILE + 2 * RADIUS; i += CTILE) {
                int ix = clampi(x0 + i - RADIUS, 0, W - 1);
                int iy = clampi(y0 + j - RADIUS, 0, H - 1);
                tile[j][i] = in[iy * W + ix];
            }
        __syncthreads();

        int x = x0 + tx, y = y0 + ty;
        if (x < W && y < H) {
            float s = 0.0f;
            #pragma unroll
            for (int fy = 0; fy < FDIM; ++fy)
                for (int fx = 0; fx < FDIM; ++fx) {
                    float fv;
                    if constexpr (UseConstant) fv = cFilter[fy * FDIM + fx];
                    else                       fv = filter[fy * FDIM + fx];
                    s += tile[ty + fy][tx + fx] * fv;
                }
            out[y * W + x] = s;
        }
    } else {
        int x = blockIdx.x * blockDim.x + tx;
        int y = blockIdx.y * blockDim.y + ty;
        if (x < W && y < H) {
            float s = 0.0f;
            for (int fy = -RADIUS; fy <= RADIUS; ++fy)
                for (int fx = -RADIUS; fx <= RADIUS; ++fx) {
                    int ix = clampi(x + fx, 0, W - 1);
                    int iy = clampi(y + fy, 0, H - 1);
                    float fv;
                    if constexpr (UseConstant) fv = cFilter[(fy + RADIUS) * FDIM + (fx + RADIUS)];
                    else                       fv = filter[(fy + RADIUS) * FDIM + (fx + RADIUS)];
                    s += in[iy * W + ix] * fv;
                }
            out[y * W + x] = s;
        }
    }
}

// L3 (고도화): separable convolution — 분리 가능한 필터를 두 1D 패스(가로→세로)로.
// 픽셀당 곱셈 FDIM*FDIM → 2*FDIM (5x5: 25→10). box blur는 (1/FDIM)*(1/FDIM)로 분리됨.
__constant__ float cSep[FDIM];
__global__ void convSepH(const float* __restrict__ in, float* __restrict__ tmp, int W, int H) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < W && y < H) {
        float s = 0.0f;
        #pragma unroll
        for (int d = -RADIUS; d <= RADIUS; ++d) s += in[y * W + clampi(x + d, 0, W - 1)] * cSep[d + RADIUS];
        tmp[y * W + x] = s;
    }
}
__global__ void convSepV(const float* __restrict__ tmp, float* __restrict__ out, int W, int H) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < W && y < H) {
        float s = 0.0f;
        #pragma unroll
        for (int d = -RADIUS; d <= RADIUS; ++d) s += tmp[clampi(y + d, 0, H - 1) * W + x] * cSep[d + RADIUS];
        out[y * W + x] = s;
    }
}

// alt (기법 데모): __grid_constant__ — 필터를 값(struct)으로 넘겨 상수 뱅크에 배치.
//   cudaMemcpyToSymbol 없이 커널 인자만으로 브로드캐스트 캐시 이용 (Ampere+; 큰 read-only
//   by-value 인자의 per-thread 로컬 복사를 피함). shared 타일 경로 재사용.
struct BoxFilter { float w[FSIZE]; };
__global__ void convGridConstFilter(const __grid_constant__ BoxFilter f,
                                    const float* __restrict__ in, float* __restrict__ out, int W, int H) {
    __shared__ float tile[CTILE + 2 * RADIUS][CTILE + 2 * RADIUS];
    int tx = threadIdx.x, ty = threadIdx.y;
    int x0 = blockIdx.x * CTILE, y0 = blockIdx.y * CTILE;
    for (int j = ty; j < CTILE + 2 * RADIUS; j += CTILE)
        for (int i = tx; i < CTILE + 2 * RADIUS; i += CTILE)
            tile[j][i] = in[clampi(y0 + j - RADIUS, 0, H - 1) * W + clampi(x0 + i - RADIUS, 0, W - 1)];
    __syncthreads();
    int x = x0 + tx, y = y0 + ty;
    if (x < W && y < H) {
        float s = 0.0f;
        #pragma unroll
        for (int fy = 0; fy < FDIM; ++fy)
            for (int fx = 0; fx < FDIM; ++fx)
                s += tile[ty + fy][tx + fx] * f.w[fy * FDIM + fx];
        out[y * W + x] = s;
    }
}

// alt (기법 데모): 큰 동적 shared 타일. BTILE² 출력을 한 블록(32×32 스레드)이 처리하며
//   (BTILE+2R)² 입력 타일을 extern __shared__ 로. BTILE=128이면 ~68KB > 48KB 이므로
//   cudaFuncAttributeMaxDynamicSharedMemorySize 옵트인 필요(런처에서 설정). carveout도 런처서.
template <int BTILE>
__global__ void convBigTile(const float* __restrict__ in, float* __restrict__ out, int W, int H) {
    extern __shared__ float bt[];                 // (BTILE+2R) × (BTILE+2R)
    const int TW = BTILE + 2 * RADIUS;
    int x0 = blockIdx.x * BTILE, y0 = blockIdx.y * BTILE;
    for (int j = threadIdx.y; j < TW; j += blockDim.y)
        for (int i = threadIdx.x; i < TW; i += blockDim.x)
            bt[j * TW + i] = in[clampi(y0 + j - RADIUS, 0, H - 1) * W + clampi(x0 + i - RADIUS, 0, W - 1)];
    __syncthreads();
    // 스레드당 (BTILE/blockDim)² 출력 처리
    for (int oy = threadIdx.y; oy < BTILE; oy += blockDim.y)
        for (int ox = threadIdx.x; ox < BTILE; ox += blockDim.x) {
            int x = x0 + ox, y = y0 + oy;
            if (x < W && y < H) {
                float s = 0.0f;
                #pragma unroll
                for (int fy = 0; fy < FDIM; ++fy)
                    for (int fx = 0; fx < FDIM; ++fx)
                        s += bt[(oy + fy) * TW + (ox + fx)] * cFilter[fy * FDIM + fx];
                out[y * W + x] = s;
            }
        }
}

// alt (compute ILP 데모): register-tiled conv. 스레드당 TO×TO 출력을 독립 누산기
//   acc[TO][TO]에 모아 FMA 파이프라인을 채운다(ILP). 출력은 인터리브 매핑(인접 스레드=인접
//   출력)이라 write coalesced. compute-bound stencil을 766 GFLOP/s 실링 쪽으로 밀어올림.
//   block=(16,16), 출력타일 OT=16*TO, shared (OT+2R)² (동적).
template <int TO>
__global__ void convRegTileILP(const float* __restrict__ in, float* __restrict__ out, int W, int H) {
    const int OT = 16 * TO;
    const int ST = OT + 2 * RADIUS;
    extern __shared__ float sh[];                 // ST × ST
    int bx = blockIdx.x * OT, by = blockIdx.y * OT;
    int tx = threadIdx.x, ty = threadIdx.y;       // 0..15
    for (int j = ty; j < ST; j += 16)
        for (int i = tx; i < ST; i += 16)
            sh[j * ST + i] = in[clampi(by + j - RADIUS, 0, H - 1) * W + clampi(bx + i - RADIUS, 0, W - 1)];
    __syncthreads();

    float acc[TO][TO];
    #pragma unroll
    for (int a = 0; a < TO; ++a)
        for (int b = 0; b < TO; ++b) acc[a][b] = 0.0f;

    #pragma unroll
    for (int fy = 0; fy < FDIM; ++fy)
        for (int fx = 0; fx < FDIM; ++fx) {
            float fv = cFilter[fy * FDIM + fx];
            #pragma unroll
            for (int a = 0; a < TO; ++a)
                for (int b = 0; b < TO; ++b)                 // TO×TO 독립 FMA = ILP
                    acc[a][b] += sh[(ty + a * 16 + fy) * ST + (tx + b * 16 + fx)] * fv;
        }
    #pragma unroll
    for (int a = 0; a < TO; ++a)
        for (int b = 0; b < TO; ++b) {
            int ox = bx + tx + b * 16, oy = by + ty + a * 16;   // 인터리브 → coalesced write
            if (ox < W && oy < H) out[oy * W + ox] = acc[a][b];
        }
}
