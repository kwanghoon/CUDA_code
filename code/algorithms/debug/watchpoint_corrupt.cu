// watchpoint_corrupt.cu
// cuda-gdb 케이스 ④: 값이 언제/누구에 의해 오염되는지 watchpoint로 포착.
// 흔한 실수: shared 인덱스 오류로 남의 슬롯을 덮어씀(aliasing write) → 조용한 오염.
// 빌드: nvcc -G -g -arch=sm_87 watchpoint_corrupt.cu -o watchpoint_corrupt
#include <cstdio>
#include <cuda_runtime.h>

// 각 스레드가 자기 슬롯에 써야 하는데, BUG로 인덱스가 겹쳐 서로의 값을 덮어씀.
__global__ void aliasWrite(const int* __restrict__ in, int* __restrict__ out, int n) {
    __shared__ int s[256];
    int t = threadIdx.x;
    s[t] = in[t];
    __syncthreads();
    // BUG: 이웃과 교환하려다 인덱스 오타 — (t^1) 대신 (t&1)로 써서 0/1 슬롯에 몰림
    int partner = (t & 1);              // 의도: t ^ 1 (짝수↔홀수 교환)
    int myval = s[t];
    __syncthreads();
    s[partner] = myval;                 // BUG: 거의 모든 스레드가 s[0]/s[1]에 몰려 덮어씀
    __syncthreads();
    out[t] = s[t];
}

int main() {
    int n = 256; int h[256], out[256];
    for (int i = 0; i < n; ++i) h[i] = i;
    int *din, *dout; cudaMalloc(&din, n * 4); cudaMalloc(&dout, n * 4);
    cudaMemcpy(din, h, n * 4, cudaMemcpyHostToDevice);
    aliasWrite<<<1, 256>>>(din, dout, n);
    cudaMemcpy(out, dout, n * 4, cudaMemcpyDeviceToHost);
    std::printf("out[10]=%d (기대: 교환값 11) → 오염됨. cuda-gdb watch로 누가 s[10] 건드리는지 추적\n", out[10]);
    return 0;
}

// cuda-gdb 진단 흐름
//   $ cuda-gdb ./watchpoint_corrupt
//   (cuda-gdb) break aliasWrite
//   (cuda-gdb) run
//   (cuda-gdb) cuda thread (10,0,0)          # 오염된 슬롯의 주인 스레드로 포커스
//   (cuda-gdb) watch s[10]                    # s[10] 이 바뀌는 순간 정지 (쓰기 감시)
//   (cuda-gdb) continue
//   → s[10] 을 쓰는 스레드에서 멈춘다.
//   (cuda-gdb) print threadIdx.x , partner    # 누가 s[10]에 썼나 / partner 값 확인
//   (cuda-gdb) print partner                  # (t&1) → 0 또는 1 이어야 하는데 s[10]?? 논리 점검
//   (cuda-gdb) print t , (t ^ 1)              # 의도한 인덱스(t^1)와 실제(t&1) 비교 → 오타 발견
//   결론: partner = t ^ 1 이어야 함. (t & 1) 오타로 슬롯 aliasing.
//   팁: rwatch(읽기)/awatch(읽기·쓰기)로도 감시 가능. 전역 주소도 `watch *(int*)0x....`.
