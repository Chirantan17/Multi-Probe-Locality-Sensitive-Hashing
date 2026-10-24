cmake_minimum_required(VERSION 3.15)
project(LSH_Engine VERSION 2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2 /O2 /W4 /EHsc")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma -O3 -Wall -Wextra")
endif()

find_package(OpenMP REQUIRED)

include_directories(backend/include)

add_executable(lsh_bench backend/src/main.cpp)
target_link_libraries(lsh_bench PRIVATE OpenMP::OpenMP_CXX)
if(WIN32)
    target_link_libraries(lsh_bench PRIVATE psapi)
endif()

enable_testing()
add_executable(test_correctness backend/tests/test_correctness.cpp)
target_link_libraries(test_correctness PRIVATE OpenMP::OpenMP_CXX)
add_test(NAME CorrectnessSuite COMMAND test_correctness)
