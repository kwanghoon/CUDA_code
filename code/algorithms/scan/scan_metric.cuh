// scan_metric.cuh
// 슬라이드: part5/chapter38 (Scan / Prefix Sum)
// scan 알고리즘의 입력 생성 / CPU 기준 / 메트릭(read+write bytes).
#pragma once

#include "../common/metrics.cuh"
#include "scan_kernels.cuh"

#include <vector>

inline std::vector<int> scanMakeInput(int N) {
    std::vector<int> v(N);
    for (int i = 0; i < N; ++i) v[i] = (i % 7) - 3;   // 음수 포함 일반 값
    return v;
}

// 세그먼트별 inclusive scan (정답)
inline std::vector<int> scanCpuReference(const std::vector<int>& in, int numSeg) {
    std::vector<int> ref(in.size());
    for (int s = 0; s < numSeg; ++s) {
        int acc = 0, base = s * SEG;
        for (int i = 0; i < SEG; ++i) { acc += in[base + i]; ref[base + i] = acc; }
    }
    return ref;
}

// scan은 memory-bound: 원소당 read+write.
inline Metric scanMetric() {
    Metric m;
    m.bytes = [](long N) { return 2.0 * N * sizeof(int); };
    return m;
}
