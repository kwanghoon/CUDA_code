# CUDA 알고리즘 최적화 예제 (Part 5)

각 알고리즘을 **naive 베이스라인 → 누적 최적화 레벨 L0..Ln** 으로 발전시키며,
레벨마다 한 기법을 추가해 성능이 단조 개선되는 과정을 실측(GB/s·GFLOP/s)한다.

## 구조
- **common/** — 재사용 헤더온리 코어 (raii · pool · variant · verify · metrics · occupancy · harness · analysis · bandwidth · cli). 모든 알고리즘이 `../common/` 로 참조.
- **reduction · scan · matmul · convolution · sort · histogram · SpMV · FFT · softmax · attention/** — 알고리즘별 클라이언트 (`*_kernels.cuh` 커널 래더, `main.cu` 벤치, `test_*.cu` GoogleTest, `CMakeLists.txt`).
- **pitfalls/** — 흔한 실수 갤러리 + `algo_traps` (알고리즘별 baseline 함정).
- **debug/** — cuda-gdb 진단 케이스 (oob / deadlock / conditional bp / watchpoint …).
- **tma/** — Hopper TMA 자동 폴백 데모.
- **practice/** — 실습용 `baseline.cu`(시작점) + `solution.cu`(정답), 맨 위 주석에 ncu 단계별 힌트.

## 빌드 (폴더별 독립)
```bash
cd FFT
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=87   # 본인 GPU: A100=80, RTX40=89, H100=90
cmake --build build -j
./build/fft_bench            # 테스트: -DBUILD_TESTING=ON 후 ctest
```
전체 한 번에: `ARCH=87 ./build_all.sh`

> roofline 실링은 GPU마다 다르다 — 성능 수치는 실행한 GPU에서 측정해 비교하라.
