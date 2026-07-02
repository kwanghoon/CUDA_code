// tma_demo.cu
// 슬라이드: part2/chapter13 (Async Copy & TMA) — TMA-capable 타일 로드 + 자동 폴백.
// cuda::memcpy_async + cuda::barrier 는 아키텍처에 따라 자동 디스패치한다:
//   Hopper(sm_90) : cp.async.bulk (TMA)   ← HW 필요, 실행 검증 불가
//   Ampere(sm_80+): cp.async               ← Orin(sm_87)에서 이 경로 실행/검증
//   그 이하        : 동기 복사
// 즉 #if 없이 폴백되며, Orin에서는 cp.async 경로가 돌아 정확성을 검증한다.
#include "../common/raii.cuh"

#include <cuda/barrier>
#include <cooperative_groups.h>
#include <cstdio>
#include <vector>

namespace cg = cooperative_groups;

constexpr int TILE = 256;   // 블록당 타일(정렬된 bulk 복사 유도)

// global→shared 를 bulk async 복사(가능하면 TMA), shared→global 로 기록(identity).
__global__ void tileCopy(const int* __restrict__ in, int* __restrict__ out, int n) {
    __shared__ alignas(16) int smem[TILE];
#pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ cuda::barrier<cuda::thread_scope_block> bar;
#pragma nv_diag_default static_var_with_dynamic_init

    auto block = cg::this_thread_block();
    if (block.thread_rank() == 0) init(&bar, block.size());
    block.sync();

    int base = blockIdx.x * TILE;
    if (base >= n) return;

    // 16B 정렬 + aligned_size_t → Hopper면 cp.async.bulk(TMA), Ampere면 cp.async
    cuda::memcpy_async(block, smem, in + base,
                       cuda::aligned_size_t<16>(TILE * sizeof(int)), bar);
    bar.arrive_and_wait();

    int t = block.thread_rank();
    out[base + t] = smem[t];
}

int main() {
    const int n = TILE * 4096;
    std::vector<int> h(n);
    for (int i = 0; i < n; ++i) h[i] = i;

    DeviceBuffer<int> d_in(n), d_out(n);
    d_in.copyFromHost(h.data());
    tileCopy<<<n / TILE, TILE>>>(d_in.data(), d_out.data(), n);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<int> out(n);
    d_out.copyToHost(out.data());
    bool ok = (out == h);

    cudaDeviceProp p{};
    CHECK_CUDA(cudaGetDeviceProperties(&p, 0));
    const char* path = (p.major >= 9) ? "TMA(cp.async.bulk)"
                     : (p.major >= 8) ? "cp.async" : "sync";
    std::printf("SM %d.%d → %s 경로.  tile copy 검증: %s\n",
                p.major, p.minor, path, ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
