// histogram_registry.cuh
// 슬라이드: part5/chapter37 (Histogram)
// 누적 최적화 레벨 L0..L3. L1~L3은 histComposed 인스턴스(플래그 1개씩 추가).
#pragma once

#include "../common/variant.cuh"
#include "histogram_kernels.cuh"

#include <cuda_runtime.h>
#include <vector>

using HistSig = void(const int*, int*, int);

static inline int ceilDiv(int a, int b) { return (a + b - 1) / b; }
static inline int cap1024(int g) { return g > 1024 ? 1024 : (g < 1 ? 1 : g); }

inline std::vector<Variant<HistSig>> makeHistogramVariants() {
    return {
        {"L0", "L0 global atomic",
            [](const int* in, int* out, int n) {
                cudaMemset(out, 0, NBINS * sizeof(int));
                kHistGlobal<<<cap1024(ceilDiv(n, HBLOCK)), HBLOCK>>>(in, out, n);
            }},
        {"L1", "L1 privatized shared",
            [](const int* in, int* out, int n) {
                cudaMemset(out, 0, NBINS * sizeof(int));
                histComposed<false, false><<<ceilDiv(n, HBLOCK), HBLOCK>>>(in, out, n);
            }},
        {"L2", "L2 +grid-stride",
            [](const int* in, int* out, int n) {
                cudaMemset(out, 0, NBINS * sizeof(int));
                histComposed<false, true><<<cap1024(ceilDiv(n, HBLOCK)), HBLOCK>>>(in, out, n);
            }},
        {"L3", "L3 +warp-agg (bit intr., cond.)",
            [](const int* in, int* out, int n) {
                cudaMemset(out, 0, NBINS * sizeof(int));
                histComposed<true, true><<<cap1024(ceilDiv(n, HBLOCK)), HBLOCK>>>(in, out, n);
            }},
        {"L4", "L4 sub-histograms (4x replica)",   // 고도화: 경합 분산 (구형 GPU서 큰 이득)
            [](const int* in, int* out, int n) {
                cudaMemset(out, 0, NBINS * sizeof(int));
                histSubHist<4><<<cap1024(ceilDiv(n, HBLOCK)), HBLOCK>>>(in, out, n);
            }},
    };
}
