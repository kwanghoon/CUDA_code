# Chapter 33 — Reduction 점진 최적화 벤치마크

합계 리덕션을 naive에서 최적화로 점진 발전시키며 측정한다. 공용 코어 `../cuda_bench`를 재사용하는
**두 번째 클라이언트**(scan=ch38 다음)로, 프레임워크가 알고리즘 간 재사용됨을 보인다.

## 스테이지
| key | 스테이지 | 기법 | warp divergence |
|-----|----------|------|-----------------|
| `atomic`      | Naive atomicAdd     | 원소마다 전역 atomic (경합 큼) | — |
| `interleaved` | Interleaved shared  | 트리 리덕션 + 블록당 atomic 1회 | 없음 |
| `warp`        | Warp-shuffle        | `__shfl_down` + 블록 부분합 | 없음 |
| `gridstride`  | Grid-stride         | SM 맞춤 grid + 스레드당 다원소 누적 | 없음 |

## 파일 (클라이언트 레시피)
`reduction_kernels.cuh` · `reduction_registry.cuh` · `reduction_metric.cuh` (bytes=read) ·
`main.cu` · `test_reduction.cu` · `analyze.cu` · `CMakeLists.txt`

## 빌드 & 실행
```bash
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=87 && cmake --build build -j
./build/reduction_bench                 # 전체 스테이지
./build/reduction_bench -a warp -a gridstride
./build/reduction_analyze 4194304 50    # Amdahl/roofline
cd build && ctest                       # GoogleTest
cmake --build build --target sanitize_racecheck
```
Metric: bytes = `N*sizeof(int)` (전량 read). verify: 단일 합 vs CPU 합계.
