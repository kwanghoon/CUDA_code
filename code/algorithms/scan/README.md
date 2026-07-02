# Chapter 38 — Scan (Prefix Sum) 점진 최적화 벤치마크

Segmented inclusive scan을 naive baseline에서 최적화 단계로 점진 발전시키며,
Part 1~3 기법을 "묶음 스테이지"로 적용하고 CUDA event로 성능을 측정한다.

## 설계

- **경량 Strategy 패턴**: 각 변형은 `std::function` 런처(`ScanLauncher`)로 등록(`scan_registry.cuh`).
  가상 클래스 계층 대신 함수 객체 테이블 — 이름으로 선택, 확장은 항목 하나 추가.
- **RAII 유틸**(`cuda_raii.cuh`): `DeviceBuffer<T>`, `GpuTimer`, `CHECK_CUDA`.
- **CLI 분리**(`cli_args.hpp`): `-n/-i/-a/-l/-h` 플래그 파싱.
- **아키텍처 폴백은 #if 없이**: `__pipeline_*` 는 sm_80 미만에서 자동 동기 폴백.
- **크기 크게**: 기본 N = 1<<24 (argv `-n` 으로 조정).

## 스테이지

| key | 스테이지 | 적용 기법 | warp divergence |
|-----|----------|-----------|-----------------|
| `seq`      | Sequential        | baseline(1 thread/seg) | 병렬성 없음 |
| `hillis`   | Hillis-Steele     | shared + ping-pong 더블버퍼 + coalesced | 없음(predication) |
| `blelloch` | Blelloch          | work-efficient + bank-conflict-free padding | 있음(`if(tid<d)`) |
| `warp`     | Warp-shuffle      | `__shfl_up_sync` + 블록 결합 | 없음 |
| `async`    | cp.async prefetch | 프리페치 더블버퍼 + grid-stride (producer-consumer) | 없음 |

## 파일

```
cuda_raii.cuh       RAII (DeviceBuffer, GpuTimer, CHECK_CUDA)
scan_kernels.cuh    커널 5종 + device 헬퍼 (cuda-gdb 주석 포함)
scan_registry.cuh   이름→런처 레지스트리 (Strategy)
scan_run.cuh        CPU 기준 스캔 + 변형 1회 실행 헬퍼
scan_benchmark.cuh  event 측정 + 검증 + 표 출력
cli_args.hpp        플래그 CLI 파서
main.cu             벤치마크 진입점
test_scan.cu        GoogleTest (모든 변형 vs CPU)
gds_demo.cu         GPUDirect Storage 데모 (옵션)
CMakeLists.txt
```

## 빌드 & 실행

```bash
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=87   # Jetson Orin
cmake --build build -j

./build/scan_bench                # 전체 스테이지, 기본 N
./build/scan_bench -n 33554432 -i 200
./build/scan_bench -a warp -a async   # 특정 스테이지만
./build/scan_bench -l                 # 변형 목록
```

## 테스트 (GoogleTest)

```bash
cd build && ctest --output-on-failure
# 또는
./build/scan_tests
```

## 정확성 / 디버깅 도구

```bash
# 메모리/레이스/동기화/초기화 검사
cmake --build build --target sanitize_memcheck
cmake --build build --target sanitize_racecheck
cmake --build build --target sanitize_synccheck
cmake --build build --target sanitize_initcheck

# cuda-gdb (디바이스 디버그 심볼 필요)
cmake -B build-dbg -DCMAKE_CUDA_FLAGS="-G -g" && cmake --build build-dbg
cuda-gdb --args ./build-dbg/scan_bench -a blelloch -n 65536
```

## GPUDirect Storage (옵션)

```bash
cmake -B build -DUSE_GDS=ON && cmake --build build --target gds_demo
./build/gds_demo data.i32       # int32 바이너리 파일을 GPU로 직접 로드 후 스캔
```
libcufile + GDS 지원 스토리지 필요. Jetson에서는 compat 모드로만 동작할 수 있다.
