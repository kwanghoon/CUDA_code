// cuda_gdb_tour.cu
// cuda-gdb 기능 투어: 관찰거리 많은 커널(shared, 루프 prefix, warp shuffle)로
// 포커스 전환/watch/조건 breakpoint/메모리 검사를 실습.
// 디버그 빌드: nvcc -G -g -arch=sm_87 cuda_gdb_tour.cu -o tour
#include "../common/raii.cuh"

#include <cstdio>
#include <vector>

__global__ void tourKernel(const int* __restrict__ in, int* __restrict__ out, int n) {
    __shared__ int s[128];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    int val = (gid < n) ? in[gid] : 0;        // (A) watch 대상
    s[tid] = val;                             // (B) shared — info/print로 확인
    __syncthreads();

    int acc = 0;
    for (int k = 0; k <= tid; ++k) acc += s[k];   // (C) 루프 — step/watch acc

    int lane = tid & 31;
    int warpSum = val;
    #pragma unroll
    for (int o = 1; o < 32; o <<= 1) {            // (D) warp shuffle — lane 전환 관찰
        int up = __shfl_up_sync(0xffffffffu, warpSum, o);
        if (lane >= o) warpSum += up;
    }

    if (gid < n) out[gid] = acc + warpSum;        // (E) 최종
}

// cuda-gdb 기능 투어 (tourKernel 을 -G -g 로 빌드 후)
//   $ cuda-gdb ./tour
//
// [breakpoint]
//   (cuda-gdb) break tourKernel                         # 함수 진입
//   (cuda-gdb) break cuda_gdb_tour.cu:20                 # 라인
//   (cuda-gdb) break tourKernel if threadIdx.x == 5      # 조건부(스레드 5만)
//   (cuda-gdb) set cuda break_on_launch application      # 모든 커널 진입에서 정지
//   (cuda-gdb) run
//
// [포커스 전환 — 어느 스레드/워프/레인을 보는지]
//   (cuda-gdb) cuda kernel block thread                  # 현재 포커스 좌표
//   (cuda-gdb) cuda thread (5,0,0)                        # 스레드 5로
//   (cuda-gdb) cuda block (0,0,0) thread (37,0,0)         # 특정 블록·스레드
//   (cuda-gdb) cuda lane 3                                # 워프 내 레인 3으로
//   (cuda-gdb) cuda sm warp                               # 현재 SM/워프
//
// [상태 조회]
//   (cuda-gdb) info cuda kernels                          # 실행 중 커널
//   (cuda-gdb) info cuda blocks                           # 블록 목록
//   (cuda-gdb) info cuda threads                          # 스레드(활성/PC) — 대량이면 필터
//   (cuda-gdb) info cuda warps                            # 워프별 PC/활성 마스크
//   (cuda-gdb) info cuda lanes                            # 워프 내 레인 상태
//   (cuda-gdb) info cuda sms                              # SM 점유
//
// [변수/메모리 검사]
//   (cuda-gdb) print val                                 # 지역 변수
//   (cuda-gdb) print s[tid]        print s[0]@8           # shared 배열 8개
//   (cuda-gdb) print/x __activemask()                     # 활성 워프 마스크
//   (cuda-gdb) print $laneid , $warpid                    # 내장 레지스터
//   (cuda-gdb) x/8dw &s[0]                                 # 메모리 examine(10진 word 8개)
//   (cuda-gdb) info registers                             # 하드웨어 레지스터
//   (cuda-gdb) set var acc = 0                             # 값 변경(실험)
//
// [watchpoint — 값이 바뀌는 순간 정지]
//   (cuda-gdb) watch acc                                  # acc 쓰기 시 정지(루프 추적)
//   (cuda-gdb) rwatch s[5]        awatch warpSum           # 읽기/읽기+쓰기 watch
//   (cuda-gdb) continue                                    # 다음 변경까지
//
// [진행 제어]
//   (cuda-gdb) next / step / finish / continue
//   (cuda-gdb) autostep 5                                  # 5줄 자동 스텝하며 관찰
//   (cuda-gdb) bt                                          # 디바이스 콜스택
//
// 관찰 포인트:
//   (C) 루프에서 `watch acc` → tid+1 번 증가하는 걸 본다.
//   (D) shuffle 후 `cuda lane N` 로 레인을 옮기며 warpSum 이 prefix 로 커지는 걸 본다.
//   조건부 break(threadIdx.x==5)로 특정 스레드만 잡아 디버깅 대상 축소.

int main() {
    const int n = 256;
    std::vector<int> h(n);
    for (int i = 0; i < n; ++i) h[i] = i % 10;

    DeviceBuffer<int> d_in(n), d_out(n);
    d_in.copyFromHost(h.data());
    tourKernel<<<(n + 127) / 128, 128>>>(d_in.data(), d_out.data(), n);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<int> out(n);
    d_out.copyToHost(out.data());
    std::printf("tour done: out[0]=%d out[5]=%d out[37]=%d out[255]=%d\n",
                out[0], out[5], out[37], out[255]);
    return 0;
}
