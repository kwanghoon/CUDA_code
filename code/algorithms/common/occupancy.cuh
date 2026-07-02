// common/occupancy.cuh
// 재사용 코어: 커널의 이론 occupancy(%) 계산.
#pragma once

#include <cuda_runtime.h>
#include "raii.cuh"

// cudaOccupancyMaxActiveBlocksPerMultiprocessor 로 이론 점유율(%) 산출.
template <typename Kernel>
double theoreticalOccupancy(Kernel kernel, int blockSize, size_t dynSmem = 0) {
    int maxBlocks = 0;
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocks, kernel, blockSize, dynSmem));
    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    int activeWarps = maxBlocks * (blockSize / 32);
    int maxWarps    = prop.maxThreadsPerMultiProcessor / 32;
    return maxWarps ? 100.0 * activeWarps / maxWarps : 0.0;
}

// 자동 런치 설정: 블록크기는 occupancy 최적으로, grid는 SM 맞춤(grid-stride 용).
//   수동 튜닝 없이 GPU마다 적응. blockSize/gridSize 를 채워 반환.
template <typename Kernel>
inline void autoLaunchConfig(Kernel kernel, int& blockSize, int& gridSize, size_t dynSmem = 0) {
    int minGrid = 0;
    CHECK_CUDA(cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize, kernel, dynSmem, 0));
    int blocksPerSM = 0;
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel, blockSize, dynSmem));
    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    gridSize = prop.multiProcessorCount * blocksPerSM;   // SM 맞춤 (grid-stride)
}
