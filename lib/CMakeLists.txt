add_library(
        lib-diffdp

        src/chart.cpp

        src/algorithm/eisner.cpp
        src/algorithm/binary_phrase.cpp

        src/dynet/eisner.cpp
        src/dynet/binary_phrase.cpp
        #src/dynet/matrix_tree_theorem.cpp

        src/builder/dependency.cpp
        src/builder/binary-phrase.cpp
)

# Define headers for this library. PUBLIC headers are used for
# compiling the library, and will be added to consumers' build
# paths.
target_include_directories(
        lib-diffdp

        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>

        PRIVATE
        src
)
add_subdirectory("/Users/filippo/repos/dynet-tools" dytools)

target_link_libraries(lib-diffdp ${Boost_LIBRARIES})
target_link_libraries(lib-diffdp dynet)
target_link_libraries(lib-diffdp libdytools)