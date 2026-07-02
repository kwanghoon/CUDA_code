cmake_minimum_required(VERSION 3.18)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 87)  # Jetson Orin sm_87 (WMMA=SM7.0+, cp.async=SM8.0+). project() 이전에 설정해야 함.
endif()
project(chapter33_reduction LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

set(CORE ${CMAKE_CURRENT_SOURCE_DIR}/../common)

foreach(exe reduction_bench:main.cu reduction_analyze:analyze.cu)
  string(REPLACE ":" ";" pair ${exe})
  list(GET pair 0 name)
  list(GET pair 1 src)
  add_executable(${name} ${src})
  target_include_directories(${name} PRIVATE ${CORE})
  target_compile_options(${name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
endforeach()

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
  add_executable(reduction_tests test_reduction.cu)
  target_include_directories(reduction_tests PRIVATE ${CORE})
  target_link_libraries(reduction_tests PRIVATE GTest::gtest_main)
  include(GoogleTest)
  gtest_discover_tests(reduction_tests)
endif()

foreach(tool memcheck racecheck synccheck initcheck)
  add_custom_target(sanitize_${tool}
    COMMAND compute-sanitizer --tool ${tool} $<TARGET_FILE:reduction_bench> -n 65536
    DEPENDS reduction_bench USES_TERMINAL)
endforeach()
