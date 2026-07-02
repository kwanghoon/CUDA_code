// common/bandwidth.cuh
// 재사용 코어: 실측 디바이스 메모리 대역폭 (grid-stride copy 마이크로벤치).
// clock 유도 이론 peak(Tegra에선 memoryClockRate 오보고로 부정확)과 별개로,
// 실제 달성 가능한 read+write copy 대역폭을 roofline 상단으로 준다.
//   주의: 이 실링은 read+write copy 기준. 단방향(read-only) 커널(예: reduction/histogram)은
//   쓰기 트래픽이 없어 이 값을 넘길 수 있다 → 그런 커널의 %peak > 100%는 버그가 아니라 정상.
#pragma once

#include <cuda_runtime.h>
#include "raii.cuh"

// float4(128-bit) 벡터화 copy — scalar copy는 트랜잭션이 작아 실제 대역폭을 과소측정한다.
__global__ void bwCopyKernel(const float4* __restrict__ in, float4* __restrict__ out, size_t n4) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
         i < n4; i += (size_t)gridDim.x * blockDim.x)
        out[i] = in[i];
}

// 측정된 디바이스 메모리 대역폭(GB/s): float4 read+write copy (달성 가능 상단).
inline double measuredBandwidthGBs(size_t nElems = (1u << 24), int iters = 50) {
    size_t n4 = nElems / 4;
    DeviceBuffer<float4> in(n4), out(n4);
    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    int block = 256;
    int grid  = prop.multiProcessorCount * 32;

    bwCopyKernel<<<grid, block>>>(in.data(), out.data(), n4);
    CHECK_CUDA(cudaDeviceSynchronize());

    GpuTimer t;
    t.start();
    for (int i = 0; i < iters; ++i) bwCopyKernel<<<grid, block>>>(in.data(), out.data(), n4);
    double ms = t.stop() / iters;

    double bytes = 2.0 * n4 * sizeof(float4);   // read + write
    return bytes / (ms / 1.0e3) / 1.0e9;
}
