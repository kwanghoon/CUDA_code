// common/analysis.cuh
// 재사용 코어: Amdahl/Gustafson + roofline(peak 대역폭) 분석.
// GPU를 H2D/kernel/D2H로 분해한 시간과 데이터 바이트로 이론 상한을 계산한다.
#pragma once

#include <cstdio>
#include <cuda_runtime.h>
#include "raii.cuh"
#include "bandwidth.cuh"

// LPDDR/GDDR peak 대역폭(GB/s) = 2(DDR) * memClk(Hz) * busWidth(byte).
inline double peakBandwidthGBs() {
    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    return 2.0 * prop.memoryClockRate * 1.0e3 * (prop.memoryBusWidth / 8.0) / 1.0e9;
}

// dataBytes = 커널이 옮기는 read+write 바이트. 전송(H2D+D2H)은 병렬화 안 되는 직렬 오버헤드.
inline void printAmdahlAnalysis(double cpuMs, double h2dMs, double kerMs, double d2hMs,
                                double dataBytes) {
    double e2e    = h2dMs + kerMs + d2hMs;
    double serial = h2dMs + d2hMs;

    std::printf("CPU                    : %8.4f ms\n", cpuMs);
    std::printf("GPU H2D (직렬)         : %8.4f ms\n", h2dMs);
    std::printf("GPU kernel (병렬)      : %8.4f ms\n", kerMs);
    std::printf("GPU D2H (직렬)         : %8.4f ms\n", d2hMs);
    std::printf("GPU end-to-end         : %8.4f ms\n\n", e2e);

    std::printf("speedup (kernel-only)  : %6.2fx  (전송 무시, 데이터 GPU 상주 시)\n", cpuMs / kerMs);
    std::printf("speedup (end-to-end)   : %6.2fx  (전송 포함)\n", cpuMs / e2e);
    std::printf("직렬(전송) 비율 f       : %6.1f%%\n", serial / e2e * 100.0);
    std::printf("Amdahl 상한             : %6.2fx  (커널→0 이어도 전송이 상한)\n\n", cpuMs / serial);

    double peak    = peakBandwidthGBs();
    double floorMs = dataBytes / (peak * 1.0e9) * 1.0e3;
    std::printf("이론 성능 (memory-bound):\n");
    std::printf("  peak 대역폭           : %8.1f GB/s\n", peak);
    std::printf("  커널 이론 하한         : %8.4f ms   (데이터/대역폭)\n", floorMs);
    std::printf("  커널 달성률            : %8.1f%% of peak\n", floorMs / kerMs * 100.0);
    std::printf("  이론 speedup 상한      : %8.2fx   (CPU/(이론커널+전송))\n", cpuMs / (floorMs + serial));

    double measBW = measuredBandwidthGBs();
    double kerBW  = dataBytes / (kerMs / 1.0e3) / 1.0e9;
    std::printf("  측정 대역폭(copy)     : %8.1f GB/s\n", measBW);
    std::printf("  커널 달성 대역폭       : %8.1f GB/s (%.1f%% of 측정)\n",
                kerBW, kerBW / measBW * 100.0);
}
