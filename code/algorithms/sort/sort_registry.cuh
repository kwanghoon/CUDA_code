// sort_registry.cuh
// 슬라이드: part5/chapter36 (Sorting)
// bitonic 정렬 레벨. 런처가 host 루프로 스테이지 커널을 구동한다.
#pragma once

#include "../common/variant.cuh"
#include "sort_kernels.cuh"

#include <cuda_runtime.h>
#include <vector>

using SortSig = void(const int*, int*, int);

// L0: 모든 (k,j) 스테이지를 전역 커널로
inline void sortGlobal(const int* in, int* out, int n) {
    cudaMemcpy(out, in, (size_t)n * sizeof(int), cudaMemcpyDeviceToDevice);
    int grid = (n + SBLOCK - 1) / SBLOCK;
    for (int k = 2; k <= n; k <<= 1)
        for (int j = k >> 1; j > 0; j >>= 1)
            bitonicStepGlobal<<<grid, SBLOCK>>>(out, j, k, n);
}

// L1: j >= SBLOCK 는 전역, j < SBLOCK tail은 한 번의 shared 커널로 병합
inline void sortShared(const int* in, int* out, int n) {
    cudaMemcpy(out, in, (size_t)n * sizeof(int), cudaMemcpyDeviceToDevice);
    int grid = (n + SBLOCK - 1) / SBLOCK;
    for (int k = 2; k <= n; k <<= 1) {
        int j = k >> 1;
        for (; j >= SBLOCK; j >>= 1) bitonicStepGlobal<<<grid, SBLOCK>>>(out, j, k, n);
        if (j > 0) bitonicSharedLowJ<<<n / SBLOCK, SBLOCK>>>(out, k, j, n);
    }
}

// L2: k <= SBLOCK 초기 phase는 로컬 정렬 한 커널로, 이후 k는 전역 + shared tail
inline void sortLocal(const int* in, int* out, int n) {
    cudaMemcpy(out, in, (size_t)n * sizeof(int), cudaMemcpyDeviceToDevice);
    int grid = (n + SBLOCK - 1) / SBLOCK;
    bitonicLocalSort<<<n / SBLOCK, SBLOCK>>>(out, n);        // k=2..SBLOCK 전부
    for (int k = 2 * SBLOCK; k <= n; k <<= 1) {
        int j = k >> 1;
        for (; j >= SBLOCK; j >>= 1) bitonicStepGlobal<<<grid, SBLOCK>>>(out, j, k, n);
        if (j > 0) bitonicSharedLowJ<<<n / SBLOCK, SBLOCK>>>(out, k, j, n);
    }
}

// radix (별도 알고리즘): LSD 4-bit × 8 패스, 병렬(warp-multisplit). 부호비트 XOR로 signed 순서화.
//   ping-pong/히스토그램 버퍼는 static으로 재사용(벤치 루프서 매 호출 malloc 방지).
inline void sortRadix(const int* in, int* out, int n) {
    static int cap = 0, histCap = 0;
    static int *dA = nullptr, *dB = nullptr, *dHist = nullptr, *dOff = nullptr;
    int numBlocks = (n + R4_BLK - 1) / R4_BLK;
    int total = R4_SIZE * numBlocks;
    if (n > cap) { if (dA) cudaFree(dA); if (dB) cudaFree(dB);
                   cudaMalloc(&dA, (size_t)n * 4); cudaMalloc(&dB, (size_t)n * 4); cap = n; }
    if (total > histCap) { if (dHist) cudaFree(dHist); if (dOff) cudaFree(dOff);
                           cudaMalloc(&dHist, (size_t)total * 4); cudaMalloc(&dOff, (size_t)total * 4); histCap = total; }
    const unsigned SIGN = 0x80000000u;
    int gridF = (n + 255) / 256; if (gridF > 1024) gridF = 1024;
    radixFlip<<<gridF, 256>>>(in, dA, n, SIGN);               // signed → unsigned 순서
    int *src = dA, *dst = dB;
    for (int pass = 0; pass < 8; ++pass) {                    // 4-bit × 8 = 32-bit
        int shift = pass * 4;
        radix4Hist<<<numBlocks, R4_BLK>>>(src, n, shift, dHist, numBlocks);
        radixScanOffsets<<<1, 256, 256 * sizeof(int)>>>(dHist, dOff, total);   // 1블록 병렬 스캔
        radix4Scatter<<<numBlocks, R4_BLK>>>(src, dst, n, shift, dOff, numBlocks);
        int* t = src; src = dst; dst = t;
    }
    radixFlip<<<gridF, 256>>>(src, out, n, SIGN);             // 되돌리기
}

inline std::vector<Variant<SortSig>> makeSortVariants() {
    return {
        {"L0", "L0 global bitonic", [](const int* in, int* out, int n) { sortGlobal(in, out, n); }},
        {"L1", "L1 +shared tail",   [](const int* in, int* out, int n) { sortShared(in, out, n); }},
        {"L2", "L2 +local sort",    [](const int* in, int* out, int n) { sortLocal(in, out, n); }},
        {"radix", "radix LSD (O(N) passes, diff. algo)", [](const int* in, int* out, int n) { sortRadix(in, out, n); }},
    };
}
