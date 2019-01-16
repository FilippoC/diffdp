cmake_minimum_required(VERSION 3.13)
set(CMAKE_CXX_STANDARD 11)
project(diffdp)
enable_testing()

find_package(Boost COMPONENTS unit_test_framework REQUIRED)
include_directories("/Users/filippo/repos/dynet")

add_subdirectory(lib)


add_executable(compare app/src/compare.cpp)
target_link_libraries(compare lib-diffdp)


file(GLOB TEST_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} test/test-*.cpp)
foreach(testSrc ${TEST_SRCS})
    get_filename_component(testName ${testSrc} NAME_WE)

    add_executable(${testName} ${testSrc})
    target_link_libraries(${testName} ${Boost_LIBRARIES} lib-diffdp dynet)

    add_test(NAME ${testName} COMMAND ${testName} )
endforeach(testSrc)
