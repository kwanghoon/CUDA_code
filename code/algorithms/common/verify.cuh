// common/verify.cuh
// 재사용 코어: GPU 출력과 CPU 기준 비교. 완전일치 / 허용오차 두 정책.
#pragma once

#include <cstddef>
#include <vector>

template <typename T>
struct VerifyResult {
    bool ok = true;
    long firstMismatch = -1;
    T    expected{};
    T    got{};
};

// 완전일치 (scan/reduction/정수)
template <typename T>
VerifyResult<T> verifyExact(const std::vector<T>& out, const std::vector<T>& ref) {
    if (out.size() != ref.size()) return {false, -1, T{}, T{}};
    for (size_t i = 0; i < ref.size(); ++i)
        if (out[i] != ref[i]) return {false, static_cast<long>(i), ref[i], out[i]};
    return {true, -1, T{}, T{}};
}

// 상대 허용오차 (float matmul/FFT/convolution)
template <typename T>
VerifyResult<T> verifyApprox(const std::vector<T>& out, const std::vector<T>& ref, double tol) {
    if (out.size() != ref.size()) return {false, -1, T{}, T{}};
    for (size_t i = 0; i < ref.size(); ++i) {
        double d = static_cast<double>(out[i]) - static_cast<double>(ref[i]);
        double m = static_cast<double>(ref[i]);
        if (d < 0) d = -d;
        if (m < 0) m = -m;
        if (d > tol * (1.0 + m)) return {false, static_cast<long>(i), ref[i], out[i]};
    }
    return {true, -1, T{}, T{}};
}
