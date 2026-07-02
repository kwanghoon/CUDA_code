cmake_minimum_required(VERSION 3.18)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 87)  # Jetson Orin sm_87 (WMMA=SM7.0+, cp.async=SM8.0+). project() 이전에 설정해야 함.
endif()
project(chapter40_spmv LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

set(CORE ${CMAKE_CURRENT_SOURCE_DIR}/../common)

add_executable(spmv_bench main.cu)
target_include_directories(spmv_bench PRIVATE ${CORE})
target_compile_options(spmv_bench PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)

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
  add_executable(spmv_tests test_spmv.cu)
  target_include_directories(spmv_tests PRIVATE ${CORE})
  target_link_libraries(spmv_tests PRIVATE GTest::gtest_main)
  include(GoogleTest)
  gtest_discover_tests(spmv_tests)
endif()

foreach(tool memcheck racecheck synccheck initcheck)
  add_custom_target(sanitize_${tool}
    COMMAND compute-sanitizer --tool ${tool} $<TARGET_FILE:spmv_bench> -n 4096
    DEPENDS spmv_bench USES_TERMINAL)
endforeach()
