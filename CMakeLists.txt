cmake_minimum_required(VERSION 3.12)
set(CMAKE_CXX_STANDARD 11)
project(diffdp)
enable_testing()

find_package(Boost COMPONENTS unit_test_framework regex serialization filesystem REQUIRED)
include_directories("/Users/filippo/repos/dynet")

add_subdirectory(lib)


add_executable(compare-phrase app/src/compare-phrase.cpp)
target_link_libraries(compare-phrase lib-diffdp)

add_executable(compare-eisner app/src/compare-eisner.cpp)
target_link_libraries(compare-eisner lib-diffdp)

add_executable(test-mtt app/src/test-mtt.cpp)
target_link_libraries(test-mtt lib-diffdp)

add_executable(test-dynet app/src/test-dynet.cpp)
target_link_libraries(test-dynet lib-diffdp)


file(GLOB TEST_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} test/test-*.cpp)
foreach(testSrc ${TEST_SRCS})
    get_filename_component(testName ${testSrc} NAME_WE)

    add_executable(${testName} ${testSrc})
    target_link_libraries(${testName} ${Boost_LIBRARIES} lib-diffdp dynet)

    add_test(NAME ${testName} COMMAND ${testName} )
endforeach(testSrc)
