#!/usr/bin/env bash
# 전체 알고리즘 벤치 빌드 (폴더별 독립 CMake). 본인 GPU arch로: ARCH=80/86/89/90
set -e
ARCH="${ARCH:-87}"
MODS="reduction scan matmul convolution sort histogram SpMV FFT softmax attention pitfalls debug tma practice"
for m in $MODS; do
  echo "=== $m ==="
  cmake -S "$m" -B "$m/build" -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DCMAKE_CUDA_ARCHITECTURES="$ARCH" >/dev/null
  cmake --build "$m/build" -j"$(nproc)"
done
echo "완료: 각 폴더 build/ 에 실행파일."
