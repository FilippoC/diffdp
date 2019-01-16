add_library(
        lib-diffdp
        src/chart.cpp
        src/algorithm/eisner.cpp
        src/dynet/eisner.cpp
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

target_link_libraries(lib-diffdp ${Boost_LIBRARIES})
target_link_libraries(lib-diffdp dynet)