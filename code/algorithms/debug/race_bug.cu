// race_bug.cu
// cuda-gdb 실습용 의도적 버그: 리덕션에서 __syncthreads() 누락 → race condition.
// 디버그 빌드: nvcc -G -g -arch=sm_87 race_bug.cu -o race_bug
#include "../common/raii.cuh"

#include <cstdio>
#include <vector>

constexpr int BLK = 256;

// [BUG] shared 로드 후 + 트리 리덕션 각 단계에 __syncthreads() 누락.
//       stride>=32 단계는 다른 워프가 쓴 s[tid+stride]를 읽으므로 동기화 필수 → 누락 시 data race.
__global__ void reduceBuggy(const int* __restrict__ in, int* __restrict__ out, int n) {
    __shared__ int s[BLK];
    int tid = threadIdx.x, i = blockIdx.x * blockDim.x + tid;
    s[tid] = (i < n) ? in[i] : 0;
    // [BUG] 여기에 __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] += s[tid + stride];
        // [BUG] 루프 안에도 __syncthreads();
    }
    if (tid == 0) atomicAdd(out, s[0]);
}

// [FIX] 각 단계에 __syncthreads() 추가 → race 제거.
__global__ void reduceFixed(const int* __restrict__ in, int* __restrict__ out, int n) {
    __shared__ int s[BLK];
    int tid = threadIdx.x, i = blockIdx.x * blockDim.x + tid;
    s[tid] = (i < n) ? in[i] : 0;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] += s[tid + stride];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, s[0]);
}

// cuda-gdb 워크스루 (reduceBuggy 를 -G -g 로 빌드한 뒤)
//
//   $ cuda-gdb ./race_bug
//   (cuda-gdb) break reduceBuggy                 # 커널 진입 중단점
//   (cuda-gdb) run
//   (cuda-gdb) info cuda kernels                 # 실행 중 커널 목록
//   (cuda-gdb) cuda block (0,0,0) thread (0,0,0) # 포커스 스레드 지정
//   (cuda-gdb) next                              # 로드 다음 줄로 (sync 없음 확인)
//   (cuda-gdb) print s[tid]                       # 내 shared 값
//   (cuda-gdb) print s[tid + 32]                  # 다른 워프가 쓸 값 — 아직 미갱신일 수 있음
//   (cuda-gdb) cuda thread (32,0,0)               # 32번 레인(다음 워프)로 포커스 이동
//   (cuda-gdb) print s[threadIdx.x]               # 그 워프는 아직 여기 도달 전 → race 근거
//   (cuda-gdb) info cuda warps                    # 워프별 PC(진행 위치)가 제각각임을 확인
//   관찰: sync 가 없어 워프마다 진행 위치가 달라, s[tid+stride] 읽을 때
//         파트너 워프가 아직 쓰지 않은 값을 읽는다.
//
// 자동 검출:
//   $ compute-sanitizer --tool racecheck ./race_bug      # WAR/RAW shared hazard 리포트
//   $ compute-sanitizer --tool synccheck  ./race_bug      # 잘못된/누락 배리어

int main() {
    const int n = BLK * 512;
    std::vector<int> h(n, 1);                 // 전부 1 → 정답 합 = n
    const int expected = n;

    DeviceBuffer<int> d_in(n), d_out(1);
    d_in.copyFromHost(h.data());

    auto run = [&](const char* name, auto launch) {
        int zero = 0;
        d_out.copyFromHost(&zero);
        launch();
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        int got = 0;
        d_out.copyToHost(&got);
        std::printf("%-6s sum=%-8d (expected %d)  %s\n",
                    name, got, expected, got == expected ? "OK" : "WRONG (race!)");
    };

    run("buggy", [&] { reduceBuggy<<<n / BLK, BLK>>>(d_in.data(), d_out.data(), n); });
    run("fixed", [&] { reduceFixed<<<n / BLK, BLK>>>(d_in.data(), d_out.data(), n); });
    return 0;
}
