// common/harness.cuh
// 재사용 코어: 공통형 벤치마크 루프 (scan/reduction 등 (const T* in, T* out) 형태).
// 형태가 다른 알고리즘(matmul/FFT)은 이 harness 대신 자기 루프를 작성한다(중복 허용).
#pragma once

#include <cstdio>
#include <string>
#include <vector>

#include "raii.cuh"
#include "variant.cuh"
#include "verify.cuh"
#include "metrics.cuh"
#include "bandwidth.cuh"   // roofline: 측정 대역폭 대비 %peak

// launch(v, d_in, d_out): 변형 실행 (numSeg/n 등 인자는 클로저가 캡처)
// verify(host_out) -> VerifyResult<T>
template <typename T, typename Sig, typename LaunchFn, typename VerifyFn>
void runBenchmark(const std::vector<Variant<Sig>>& variants,
                  const std::vector<T>& h_in, size_t outCount,
                  const Metric& metric, long metricN, int iters,
                  LaunchFn launch, VerifyFn verify) {
    int N = static_cast<int>(h_in.size());
    DeviceBuffer<T> d_in(N), d_out(outCount);
    d_in.copyFromHost(h_in.data());

    // roofline 기준: float4 read+write copy 대역폭(측정). read-only/ L2재사용 커널은
    //   이 값을 넘을 수 있어 %copyBW > 100%가 정상 (bandwidth.cuh 주석 참고).
    double peakBW = metric.hasBytes() ? measuredBandwidthGBs() : 0.0;
    if (peakBW > 0.0) std::printf("copy BW(read+write, float4) 기준 = %.1f GB/s\n", peakBW);

    std::printf("%-34s %10s %12s %8s %8s %10s\n",
                "Stage", "Time(ms)", "GB/s", "%copyBW", "Check", "Speedup");
    std::printf("%s\n", std::string(86, '-').c_str());

    std::vector<T> h_out(outCount);
    double baseMs = -1.0;
    for (const auto& v : variants) {
        launch(v, d_in.data(), d_out.data());
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        d_out.copyToHost(h_out.data());
        VerifyResult<T> vr = verify(h_out);
        if (!vr.ok) std::printf("  [mismatch @%ld]\n", vr.firstMismatch);

        GpuTimer t;
        t.start();
        for (int i = 0; i < iters; ++i) launch(v, d_in.data(), d_out.data());
        double ms = t.stop() / iters;

        double gbps = metric.hasBytes() ? Metric::gbPerSec(metric.bytes(metricN), ms) : 0.0;
        double pk   = (peakBW > 0.0) ? gbps / peakBW * 100.0 : 0.0;
        if (baseMs < 0) baseMs = ms;
        std::printf("%-34s %10.4f %12.2f %7.1f%% %8s %9.2fx\n",
                    v.label.c_str(), ms, gbps, pk, vr.ok ? "OK" : "FAIL", baseMs / ms);
    }
}
