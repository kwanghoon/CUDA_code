// oob_fault.cu
// cuda-gdb 케이스 ①: illegal address를 잡아 원인 스레드/인덱스 찾기.
// 흔한 실수: 경계 검사(i < n) 누락 → 남는 스레드가 out[i] 범위 밖 기록.
// 빌드: nvcc -G -g -arch=sm_87 oob_fault.cu -o oob_fault
#include <cstdio>
#include <cuda_runtime.h>

// BUG: i < n 검사가 없다 → i >= n 인 스레드가 out[i] 밖(OOB)에 쓴다.
__global__ void scaleOOB(const float* __restrict__ in, float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float v = in[i] * 2.0f;        // (i>=n 이면) 여기 in[i] 도 OOB 읽기
    out[i] = v;                    // OOB 쓰기 → CUDA_EXCEPTION_14 (Warp Illegal Address)
}

int main() {
    int n = 1000;
    float *din, *dout;
    cudaMalloc(&din, n * sizeof(float));
    cudaMalloc(&dout, n * sizeof(float));
    int block = 256;
    int grid = n;                  // BUG: (n+block-1)/block 이어야 하는데 grid=n
                                   //   → n*256 스레드가 out[i] 훨씬 밖까지 씀 → 확실한 illegal address
    scaleOOB<<<grid, block>>>(din, dout, n);
    cudaError_t e = cudaDeviceSynchronize();
    std::printf("kernel status: %s\n", cudaGetErrorString(e));
    // 주의: 개별(discrete) GPU면 보통 "an illegal memory access" 로 즉시 실패한다.
    //   Jetson(통합 메모리)은 OOB가 유효 물리 페이지에 떨어져 'no error'로 조용히 오염될 수 있다.
    //   → 어느 쪽이든 확실히 잡으려면: compute-sanitizer --tool memcheck ./oob_fault
    //     (ERROR SUMMARY 에 잡힘. 상세 per-thread 정보는 GPU 디버깅 기능 필요.)
    return 0;
}

// cuda-gdb 진단 흐름 (이 버그를 '어떻게 찾는가')
//   $ cuda-gdb ./oob_fault
//   (cuda-gdb) set cuda memcheck on          # 첫 잘못된 접근에서 즉시 정지
//   (cuda-gdb) run
//   → CUDA_EXCEPTION_14, "Warp Illegal Address" 로 멈춘다.
//   (cuda-gdb) info cuda kernels             # 어느 커널에서 났는지
//   (cuda-gdb) bt                            # scaleOOB 프레임 확인
//   (cuda-gdb) cuda thread                   # 현재 포커스(예: block 3, thread ...)
//   (cuda-gdb) print i                       # → 1000 이상 (범위 밖!)
//   (cuda-gdb) print n                       # 1000 — i>=n 인데 접근함이 드러남
//   (cuda-gdb) info cuda warps               # 마지막 블록의 어떤 워프가 예외 상태인지
//   (cuda-gdb) print threadIdx.x , blockIdx.x
//   결론: 경계 검사 `if (i < n)` 누락. 고치면: if (i < n) { out[i] = in[i]*2; }
//   (참고) 커맨드라인으로도: compute-sanitizer --tool memcheck ./oob_fault
