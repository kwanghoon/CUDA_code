// scan_registry.cuh
// 슬라이드: part5/chapter38 (Scan)
// 누적 최적화 레벨 L0..L3. L1/L2는 scanComposed<WarpShuffle> 조합, L0/L3은 별도 커널.
#pragma once

#include "../common/variant.cuh"
#include "scan_kernels.cuh"

#include <vector>

using ScanSig = void(const int*, int*, int);

inline std::vector<Variant<ScanSig>> makeScanVariants() {
    return {
        {"L0", "L0 sequential",
            [](const int* in, int* out, int numSeg) {
                int b = 256;
                kSequential<<<(numSeg + b - 1) / b, b>>>(in, out, numSeg);
            }},
        {"L1", "L1 Hillis-Steele (shared)",
            [](const int* in, int* out, int numSeg) {
                scanComposed<false><<<numSeg, SEG>>>(in, out);
            }},
        {"L2", "L2 warp-shuffle",
            [](const int* in, int* out, int numSeg) {
                scanComposed<true><<<numSeg, SEG>>>(in, out);
            }},
        {"L3", "L3 +cp.async prefetch",
            [](const int* in, int* out, int numSeg) {
                int g = numSeg < ASYNC_GRID_CAP ? numSeg : ASYNC_GRID_CAP;
                kAsyncPrefetch<<<g, SEG>>>(in, out, numSeg);
            }},
        {"L4", "L4 int4 +coarsening",
            [](const int* in, int* out, int numSeg) {
                kScanVec4<<<numSeg, VEC_THREADS>>>(in, out);   // SEG/4 스레드
            }},
    };
}
