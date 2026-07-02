cmake_minimum_required(VERSION 3.18)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 87)  # Jetson Orin sm_87 (WMMA=SM7.0+, cp.async=SM8.0+). project() 이전에 설정해야 함.
endif()
project(chapter38_scan LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

set(CORE ${CMAKE_CURRENT_SOURCE_DIR}/../common)

foreach(exe scan_bench:main.cu scan_analyze:analyze.cu pool_bench:pool_bench.cu stream_bench:stream_bench.cu unified_bench:unified_bench.cu)
  string(REPLACE ":" ";" pair ${exe})
  list(GET pair 0 name)
  list(GET pair 1 src)
  add_executable(${name} ${src})
  target_include_directories(${name} PRIVATE ${CORE})
  target_compile_options(${name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
endforeach()

# GoogleTest 단위 테스트 (기본 ON, 끄려면 -DBUILD_TESTING=OFF)
option(BUILD_TESTING "Build GoogleTest unit tests" ON)
if(BUILD_TESTING)
  find_package(GTest QUIET)
  if(NOT GTest_FOUND)
    include(FetchContent)
    FetchContent_Declare(googletest
      GIT_REPOSITORY https://github.com/google/googletest.git
      GIT_TAG        v1.17.0)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
  endif()
  enable_testing()
  add_executable(scan_tests test_scan.cu)
  target_include_directories(scan_tests PRIVATE ${CORE})
  target_link_libraries(scan_tests PRIVATE GTest::gtest_main)
  include(GoogleTest)
  gtest_discover_tests(scan_tests)
endif()

# compute-sanitizer 타겟 (make sanitize_racecheck 등)
foreach(tool memcheck racecheck synccheck initcheck)
  add_custom_target(sanitize_${tool}
    COMMAND compute-sanitizer --tool ${tool} $<TARGET_FILE:scan_bench> -n 65536
    DEPENDS scan_bench USES_TERMINAL)
endforeach()

# GPUDirect Storage 데모 (옵션: -DUSE_GDS=ON, libcufile 필요; Jetson은 compat 모드일 수 있음)
option(USE_GDS "Build GPUDirect Storage (cuFile) demo" OFF)
if(USE_GDS)
  find_library(CUFILE_LIB cufile
    HINTS /usr/local/cuda/lib64 /usr/local/cuda/targets/aarch64-linux/lib
          /usr/local/cuda/targets/x86_64-linux/lib)
  if(NOT CUFILE_LIB)
    message(FATAL_ERROR "libcufile not found — GPUDirect Storage unavailable here")
  endif()
  add_executable(gds_demo gds_demo.cu)
  target_include_directories(gds_demo PRIVATE ${CORE})
  target_link_libraries(gds_demo PRIVATE ${CUFILE_LIB})
endif()
