// sync_deadlock.cu
// cuda-gdb 케이스 ②: __syncthreads()를 발산 분기 안에 둬서 생기는 교착(hang).
// 일부 스레드만 배리어 도달 → 나머지 영원히 대기. (이 프로그램은 멈춘다 — 관찰용)
// 빌드: nvcc -G -g -arch=sm_87 sync_deadlock.cu -o sync_deadlock
#include <cstdio>
#include <cuda_runtime.h>

// BUG: 절반의 스레드만 __syncthreads() 에 도달 → 블록 전체가 배리어에서 교착.
__global__ void halfBarrier(int* out, int n) {
    __shared__ int s[256];
    int t = threadIdx.x;
    s[t] = t;
    if (t < 128) {                 // 발산 분기
        __syncthreads();           // BUG: t>=128 스레드는 여기 안 옴 → 배리어 영원히 안 채워짐
        out[blockIdx.x * blockDim.x + t] = s[t] + s[t + 1];
    }
    // 올바른 형태: __syncthreads() 는 분기 밖(모든 스레드 도달)에 둬야 한다.
}

int main() {
    int n = 256; int* dout; cudaMalloc(&dout, n * sizeof(int));
    std::printf("launching (곧 hang — cuda-gdb로 Ctrl+C 후 관찰)\n");
    halfBarrier<<<1, 256>>>(dout, n);
    cudaDeviceSynchronize();       // 여기서 영원히 반환 안 함
    std::printf("done\n");         // 도달 못 함
    return 0;
}

// cuda-gdb 진단 흐름
//   $ cuda-gdb ./sync_deadlock
//   (cuda-gdb) run
//   → 멈춘 것처럼 보이면 Ctrl+C 로 인터럽트.
//   (cuda-gdb) info cuda warps        # 워프별 PC — 워프마다 다른 위치에서 멈춰 있음
//   (cuda-gdb) info cuda lanes        # 활성 레인 마스크: 일부 레인만 배리어 앞
//   (cuda-gdb) cuda thread (0,0,0)    # t<128 스레드 → __syncthreads 라인에서 대기
//   (cuda-gdb) cuda thread (200,0,0)  # t>=128 스레드 → 배리어를 '지나치는' 경로에 없음
//   (cuda-gdb) bt                     # 두 그룹의 콜스택/PC가 갈린 것을 확인
//   결론: 배리어가 발산 분기 안에 있음. __syncthreads()를 if 밖으로 빼야 함.
//   (참고) compute-sanitizer --tool synccheck ./sync_deadlock 로도 배리어 오류 진단.
