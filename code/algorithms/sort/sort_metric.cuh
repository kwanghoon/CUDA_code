// sort_metric.cuh
// 슬라이드: part5/chapter36 (Sorting)
// 입력 생성 / CPU 기준(std::sort) / 메트릭(rough: N*4 read-write 근사).
#pragma once

#include "../common/metrics.cuh"

#include <algorithm>
#include <vector>

inline std::vector<int> sortMakeInput(int n) {
    std::vector<int> v(n);
    unsigned s = 123456789u;
    for (int i = 0; i < n; ++i) { s = s * 1103515245u + 12345u; v[i] = (int)((s >> 16) & 0xffff); }
    return v;
}

inline std::vector<int> sortCpuReference(std::vector<int> v) {
    std::sort(v.begin(), v.end());
    return v;
}

// bitonic은 다중 패스라 GB/s는 대략치(원소당 read+write 1회 기준).
inline Metric sortMetric() {
    Metric m;
    m.bytes = [](long N) { return 2.0 * N * sizeof(int); };
    return m;
}
