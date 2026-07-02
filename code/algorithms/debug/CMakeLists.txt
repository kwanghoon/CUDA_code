cmake_minimum_required(VERSION 3.18)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 87)  # Jetson Orin sm_87 (WMMA=SM7.0+, cp.async=SM8.0+). project() 이전에 설정해야 함.
endif()
project(debug_cudagdb LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

set(CORE ${CMAKE_CURRENT_SOURCE_DIR}/../common)

# 일반 빌드: buggy vs fixed 결과 비교 (race로 buggy가 종종 WRONG)
add_executable(race_bug race_bug.cu)
target_include_directories(race_bug PRIVATE ${CORE})

# 디버그 빌드: cuda-gdb 용 디바이스 심볼(-G -g). 스텝/shared 검사 가능.
add_executable(race_bug_dbg race_bug.cu)
target_include_directories(race_bug_dbg PRIVATE ${CORE})
target_compile_options(race_bug_dbg PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G -g>)

# cuda-gdb 기능 투어 (관찰거리 풍부). 디버그 심볼로 빌드.
add_executable(tour cuda_gdb_tour.cu)
target_include_directories(tour PRIVATE ${CORE})
target_compile_options(tour PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G -g>)

# cuda-gdb 진단 케이스 모음 (각각 특정 버그를 cuda-gdb로 찾는 흐름을 주석에 담음). 모두 -G -g.
#   oob_fault         : 경계검사 누락 → illegal address (set cuda memcheck on)
#   sync_deadlock     : 발산 분기 내 __syncthreads → 교착 (info cuda warps/lanes; 실행 시 hang)
#   conditional_hunt  : 한 스레드만 오염 → 조건부 breakpoint 저격
#   watchpoint_corrupt: shared aliasing 쓰기 → watch로 오염 순간 포착
foreach(case oob_fault sync_deadlock conditional_hunt watchpoint_corrupt)
  add_executable(${case} ${case}.cu)
  target_include_directories(${case} PRIVATE ${CORE})
  target_compile_options(${case} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G -g>)
endforeach()

# 편의 타겟: racecheck / synccheck / memcheck
add_custom_target(racecheck
  COMMAND compute-sanitizer --tool racecheck $<TARGET_FILE:race_bug>
  DEPENDS race_bug USES_TERMINAL)
add_custom_target(synccheck
  COMMAND compute-sanitizer --tool synccheck $<TARGET_FILE:sync_deadlock>
  DEPENDS sync_deadlock USES_TERMINAL)
add_custom_target(memcheck
  COMMAND compute-sanitizer --tool memcheck $<TARGET_FILE:oob_fault>
  DEPENDS oob_fault USES_TERMINAL)
