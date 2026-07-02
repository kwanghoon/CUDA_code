cmake_minimum_required(VERSION 3.18)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 87)  # Jetson Orin sm_87 (WMMA=SM7.0+, cp.async=SM8.0+). project() 이전에 설정해야 함.
endif()
project(chapter36_sort LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit REQUIRED)
set(CORE ${CMAKE_CURRENT_SOURCE_DIR}/../common)

add_executable(sort_bench main.cu)
target_include_directories(sort_bench PRIVATE ${CORE})
target_compile_options(sort_bench PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)

# Dynamic Parallelism 데모: separable compilation + cudadevrt 링크
find_library(CUDADEVRT_LIB cudadevrt
  HINTS ${CUDAToolkit_LIBRARY_DIR} /usr/local/cuda/lib64
        /usr/local/cuda/targets/aarch64-linux/lib /usr/local/cuda/targets/x86_64-linux/lib)
add_executable(quicksort_dp quicksort_dp.cu)
target_include_directories(quicksort_dp PRIVATE ${CORE})
set_target_properties(quicksort_dp PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(quicksort_dp PRIVATE ${CUDADEVRT_LIB})

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
  add_executable(sort_tests test_sort.cu)
  target_include_directories(sort_tests PRIVATE ${CORE})
  target_link_libraries(sort_tests PRIVATE GTest::gtest_main)
  include(GoogleTest)
  gtest_discover_tests(sort_tests)
endif()

foreach(tool memcheck racecheck synccheck initcheck)
  add_custom_target(sanitize_${tool}
    COMMAND compute-sanitizer --tool ${tool} $<TARGET_FILE:sort_bench> -n 65536
    DEPENDS sort_bench USES_TERMINAL)
endforeach()
