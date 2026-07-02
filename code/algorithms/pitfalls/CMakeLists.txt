cmake_minimum_required(VERSION 3.18)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 87)  # Jetson Orin sm_87 (WMMA=SM7.0+, cp.async=SM8.0+). project() 이전에 설정해야 함.
endif()
project(pitfalls LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

set(CORE ${CMAKE_CURRENT_SOURCE_DIR}/../common)

add_executable(pitfalls_bench pitfalls_bench.cu)
target_include_directories(pitfalls_bench PRIVATE ${CORE})
target_compile_options(pitfalls_bench PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)

# 알고리즘별 베이스라인 흔한 실수 (정확성 함정 — Check=FAIL 대비)
add_executable(algo_traps algo_traps.cu)
target_include_directories(algo_traps PRIVATE ${CORE})

# racecheck 편의 타겟 (reduction/histogram/matmul BUG는 레이스)
add_custom_target(traps_racecheck
  COMMAND compute-sanitizer --tool racecheck $<TARGET_FILE:algo_traps>
  DEPENDS algo_traps USES_TERMINAL)
