// histogram_metric.cuh
// 슬라이드: part5/chapter37 (Histogram)
// 입력 생성 / CPU 기준(히스토그램) / 메트릭(read bytes).
#pragma once

#include "../common/metrics.cuh"
#include "histogram_kernels.cuh"

#include <vector>

// 워프별로 같은 bin에 몰리는 분포 (i>>5) — warp-aggregation 효과가 잘 드러남.
inline std::vector<int> histMakeInput(int N) {
    std::vector<int> v(N);
    for (int i = 0; i < N; ++i) v[i] = (i >> 5) % NBINS;
    return v;
}

inline std::vector<int> histCpuReference(const std::vector<int>& in) {
    std::vector<int> h(NBINS, 0);
    for (int x : in) h[x & (NBINS - 1)]++;
    return h;
}

// 입력 전량 read.
inline Metric histMetric() {
    Metric m;
    m.bytes = [](long N) { return static_cast<double>(N) * sizeof(int); };
    return m;
}
