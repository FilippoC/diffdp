cmake_minimum_required(VERSION 3.13)
project(diffdp)

set(CMAKE_CXX_STANDARD 11)

add_subdirectory(lib)

add_executable(diffdp app/src/check-grad.cpp)

target_link_libraries(diffdp lib-diffdp)