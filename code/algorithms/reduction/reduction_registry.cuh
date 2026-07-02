// reduction_registry.cuh
// 누적 최적화 레벨 L0..L4. 각 레벨 = 이전 + policy 플래그 1개 (reduceComposed 인스턴스).
#pragma once

#include "../common/variant.cuh"
#include "reduction_kernels.cuh"

#include <cuda_runtime.h>
#include <vector>

using ReduceSig = void(const int*, int*, int);

static inline int ceilDiv(int a, int b) { return (a + b - 1) / b; }
static inline int cap1024(int g) { return g > 1024 ? 1024 : (g < 1 ? 1 : g); }

inline std::vector<Variant<ReduceSig>> makeReductionVariants() {
    return {
        {"L0", "L0 divergent tree",
            [](const int* in, int* out, int n) {
                cudaMemset(out, 0, sizeof(int));
                reduceComposed<false, false, false, false><<<ceilDiv(n, RBLOCK), RBLOCK>>>(in, out, n);
            }},
        {"L1", "L1 +sequential addressing",
            [](const int* in, int* out, int n) {
                cudaMemset(out, 0, sizeof(int));
                reduceComposed<true, false, false, false><<<ceilDiv(n, RBLOCK), RBLOCK>>>(in, out, n);
            }},
        {"L2", "L2 +warp-tail (sync-free)",
            [](const int* in, int* out, int n) {
                cudaMemset(out, 0, sizeof(int));
                reduceComposed<true, true, false, false><<<ceilDiv(n, RBLOCK), RBLOCK>>>(in, out, n);
            }},
        {"L3", "L3 +grid-stride",
            [](const int* in, int* out, int n) {
                cudaMemset(out, 0, sizeof(int));
                reduceComposed<true, true, true, false><<<cap1024(ceilDiv(n, RBLOCK)), RBLOCK>>>(in, out, n);
            }},
        {"L4", "L4 +int4 vectorized",
            [](const int* in, int* out, int n) {
                cudaMemset(out, 0, sizeof(int));
                reduceComposed<true, true, true, true><<<cap1024(ceilDiv(n / 4, RBLOCK)), RBLOCK>>>(in, out, n);
            }},
        // 아래 3개는 누적 사다리(L0→L4)가 아니라 L4 위의 "대체 메커니즘" 데모
        //    (같이 못 쌓임: threadfence vs grid.sync vs HW-warpreduce 중 택1). 모두 ≈L4(cond.).
        {"A1", "alt: single-pass __threadfence (on L4, cond.)",   // Orin은 L4 atomic이 이미 싸 ≈L4; 타 HW서 이득
            [](const int* in, int* out, int n) {
                cudaMemset(out, 0, sizeof(int));
                reduceSinglePass<<<cap1024(ceilDiv(n / 4, RBLOCK)), RBLOCK>>>(in, out, n);
            }},
        {"A2", "alt: coop-groups grid.sync (on L4, cond.)",   // 협력 런치, grid.sync() 단일패스
            [](const int* in, int* out, int n) {
                cudaMemset(out, 0, sizeof(int));
                // 협력 런치: 모든 블록 동시 상주 필요 → occupancy로 그리드 상한 결정
                int blocksPerSM = 0;
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, reduceCoopGroups, RBLOCK, 0);
                cudaDeviceProp p{}; cudaGetDeviceProperties(&p, 0);
                int grid = p.multiProcessorCount * blocksPerSM;
                int want = cap1024(ceilDiv(n / 4, RBLOCK));
                if (grid > want) grid = want;
                void* args[] = {(void*)&in, (void*)&out, (void*)&n};
                cudaLaunchCooperativeKernel((void*)reduceCoopGroups, grid, RBLOCK, args, 0, 0);
            }},
        {"A3", "alt: HW warp-reduce __reduce_add_sync (on L4, cond.)",   // Ampere+ 단일 명령 워프리듀스
            [](const int* in, int* out, int n) {
                cudaMemset(out, 0, sizeof(int));
                reduceHWWarp<<<cap1024(ceilDiv(n / 4, RBLOCK)), RBLOCK>>>(in, out, n);
            }},
    };
}
