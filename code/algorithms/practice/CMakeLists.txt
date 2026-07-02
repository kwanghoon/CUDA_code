cmake_minimum_required(VERSION 3.18)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 87)  # 본인 GPU arch로: 80(A100)/86/89(RTX4090)/90(H100). project() 이전.
endif()
project(part5_algorithms_practice LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

# 각 알고리즘: baseline(시작점) + solution(참고답안) 한 쌍
foreach(pair reduction/baseline reduction/solution matmul/baseline matmul/solution nbody/baseline nbody/solution)
  string(REPLACE "/" "_" name ${pair})
  add_executable(${name} ${pair}.cu)
  target_compile_options(${name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
endforeach()

# ncu 프로파일 편의 타겟 (baseline의 병목을 full set으로)
add_custom_target(ncu_reduction
  COMMAND ncu --set full $<TARGET_FILE:reduction_baseline>
  DEPENDS reduction_baseline USES_TERMINAL)
add_custom_target(ncu_matmul
  COMMAND ncu --set full $<TARGET_FILE:matmul_baseline>
  DEPENDS matmul_baseline USES_TERMINAL)
