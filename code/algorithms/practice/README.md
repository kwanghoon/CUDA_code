# Part 5 알고리즘 최적화 실습 (baseline → solution)

각 알고리즘마다 **`baseline.cu`**(시작점)와 **`solution.cu`**(완전 최적화 참고답안) 한 쌍.
실습 방식: baseline에서 시작해 **ncu로 병목을 찾아 한 기법씩** 적용하며 solution 수준까지 끌어올린다.

## 흐름
1. baseline을 빌드·실행해 기준 성능(GB/s 또는 GFLOP/s)을 잰다.
2. **각 파일 맨 위 주석의 "ncu로 무엇을 보나" 단계**를 따라간다:
   - `ncu --set full ./baseline` (또는 `make ncu_reduction` / `make ncu_matmul`)
   - 안내된 metric이 병목을 가리키면 → 대응 기법을 직접 적용
3. 한 기법 적용 → 다시 측정 → 개선 확인 → 다음 단계. 반복.
4. `solution.cu`와 성능/구현을 비교.

## 두 가지 bound 유형
- **reduction** = memory-bound: 최종 상한은 `dram__throughput`(대역폭). divergence→sequential,
  warp-tail, grid-stride, **int4 벡터화**로 대역폭에 붙인다.
- **matmul** = compute-bound: 상한은 `sm__throughput`(FP32 연산). 재사용(**shared tiling**)→
  **register tiling(ILP)**로 연산 파이프를 채운다. (더 = Tensor Core/WMMA, 심화)

## 빌드
```bash
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=87   # 본인 GPU arch (A100=80, RTX4090=89, H100=90)
cmake --build build -j
./build/reduction_baseline   ./build/reduction_solution
./build/matmul_baseline      ./build/matmul_solution
```

> roofline 실링은 GPU마다 다르다. baseline/solution 모두 실행 GPU에서 측정해 비교하라
> (한 GPU의 결과를 일반화하지 말 것). solution이 그 GPU의 bound(대역폭/FP32 peak)에
> 얼마나 근접하는지가 진짜 목표다.
