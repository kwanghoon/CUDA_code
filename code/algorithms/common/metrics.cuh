// common/metrics.cuh
// 재사용 코어: 알고리즘별 처리량 메트릭 정책.
// bytes/flops 는 클라이언트가 주입 (scan=read+write bytes, reduction=read bytes, matmul=flops).
#pragma once

#include <functional>

struct Metric {
    std::function<double(long)> bytes;   // 한 번 실행의 read+write 바이트 (null 가능)
    std::function<double(long)> flops;   // 한 번 실행의 FLOP 수 (null 가능)

    bool hasBytes() const { return static_cast<bool>(bytes); }
    bool hasFlops() const { return static_cast<bool>(flops); }

    static double gbPerSec(double b, double ms)    { return b / (ms / 1.0e3) / 1.0e9; }
    static double gflopPerSec(double f, double ms) { return f / (ms / 1.0e3) / 1.0e9; }
};
