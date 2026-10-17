cmake_minimum_required(VERSION 3.20)
project(LocalitySensitiveHashing LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

# Enable AVX2 SIMD, OpenMP, and O3 release optimizations
if (MSVC)
    add_compile_options(/arch:AVX2 /O2 /openmp)
else()
    add_compile_options(-mavx2 -mfma -O3 -fopenmp -Wall -Wextra)
    add_link_options(-fopenmp)
endif()

include_directories(backend/include)

add_executable(lsh_bench
    backend/src/main.cpp
)

target_link_libraries(lsh_bench PRIVATE OpenMP::OpenMP_CXX)