cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

# for msys2 on windows add -mconsole to CMAKE_CXX_FLAGS
set(CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS} -std=c++11 -fno-exceptions -march=native")
set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS} -g -DDEBUG -O0")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3 -fopt-info-vec-optimized -fopenmp -fopenmp-simd -fno-math-errno")

include_directories(".")
file(GLOB SOURCE_FILES  "./rt.cpp")

add_executable(path_trace ${SOURCE_FILES})
target_link_libraries(path_trace)
