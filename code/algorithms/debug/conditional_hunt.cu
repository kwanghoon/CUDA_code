// conditional_hunt.cu
// cuda-gdb 케이스 ③: 한 스레드에서만 틀리는 버그(off-by-one)를 조건부 breakpoint로 저격.
// 빌드: nvcc -G -g -arch=sm_87 conditional_hunt.cu -o conditional_hunt
#include <cstdio>
#include <cuda_runtime.h>

// 이웃 평균. BUG: 마지막 유효 인덱스에서 out[i] = (in[i]+in[i+1])/2 인데 i+1 이 다음 세그먼트로 샘.
// 여기서는 '틀린 값'이 되도록: 세그먼트 경계(i % SEG == SEG-1)에서 다음 세그먼트 값을 잘못 섞음.
#define SEG 32
__global__ void neighborAvg(const int* __restrict__ in, int* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int right = in[i + 1 < n ? i + 1 : i];   // 겉보기엔 안전
    // BUG: 세그먼트 경계에서 오른쪽 이웃을 '같은 세그먼트로 클램프'했어야 하는데 안 함
    //   → i % SEG == SEG-1 인 스레드만 다음 세그먼트 값을 섞어 결과 오염.
    out[i] = (in[i] + right) / 2;
}

int main() {
    int n = 256; int h[256], ref[256], out[256];
    for (int i = 0; i < n; ++i) h[i] = (i / SEG) * 100 + (i % SEG);   // 세그먼트별로 값 대역이 다름
    // CPU 기준: 오른쪽 이웃을 같은 세그먼트로 클램프
    for (int i = 0; i < n; ++i) {
        int seg = i / SEG, rightIdx = (i % SEG == SEG - 1) ? i : i + 1;
        ref[i] = (h[i] + h[rightIdx]) / 2;
    }
    int *din, *dout; cudaMalloc(&din, n * 4); cudaMalloc(&dout, n * 4);
    cudaMemcpy(din, h, n * 4, cudaMemcpyHostToDevice);
    neighborAvg<<<1, 256>>>(din, dout, n);
    cudaMemcpy(out, dout, n * 4, cudaMemcpyDeviceToHost);
    int bad = -1;
    for (int i = 0; i < n; ++i) if (out[i] != ref[i]) { bad = i; break; }
    if (bad < 0) std::printf("all OK\n");
    else std::printf("첫 오차 인덱스 = %d (out=%d, ref=%d) → cuda-gdb로 이 스레드만 저격\n", bad, out[bad], ref[bad]);
    return 0;
}

// cuda-gdb 진단 흐름 (위 출력이 알려준 bad 인덱스 = 31 이라고 하자)
//   $ cuda-gdb ./conditional_hunt
//   (cuda-gdb) break neighborAvg if (blockIdx.x*blockDim.x+threadIdx.x) == 31   # 그 스레드만 정지
//   (cuda-gdb) run
//   (cuda-gdb) print i                     # 31 (세그먼트 경계 SEG-1)
//   (cuda-gdb) next                         # right 계산 라인까지 진행
//   (cuda-gdb) print right                  # in[32] = 다음 세그먼트 값(100대) — 오염 원인!
//   (cuda-gdb) print in[i]  print in[i+1]   # 대역이 다름을 눈으로 확인
//   (cuda-gdb) print i % 32                 # 31 → 경계에서 클램프 누락이 드러남
//   결론: 경계에서 rightIdx 를 같은 세그먼트로 클램프해야 함.
//   팁: 조건부 bp는 수천 스레드 중 문제의 하나만 멈추게 해주는 핵심 기법.
