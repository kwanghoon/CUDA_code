// reduction_metric.cuh
// 리덕션 입력 생성 / CPU 기준(합) / 메트릭(read bytes).
#pragma once

#include "../common/metrics.cuh"

#include <vector>

inline std::vector<int> reduceMakeInput(int N) {
    std::vector<int> v(N);
    for (int i = 0; i < N; ++i) v[i] = (i % 7) - 3;   // 7주기 합=0, 결과 작음 → int 안전
    return v;
}

// 전체 합 (정답). 결과는 크기 1 벡터 (harness out[0]과 비교).
inline std::vector<int> reduceCpuReference(const std::vector<int>& in) {
    long s = 0;
    for (int x : in) s += x;
    return { static_cast<int>(s) };
}

// 리덕션은 memory-bound: 원소당 read 1회 (write는 소량).
inline Metric reduceMetric() {
    Metric m;
    m.bytes = [](long N) { return static_cast<double>(N) * sizeof(int); };
    return m;
}
