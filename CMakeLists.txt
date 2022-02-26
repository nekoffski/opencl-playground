cmake_minimum_required(VERSION 3.16)
project(opencl-playground)

add_definitions(-DSPDLOG_COMPILED_LIB=1)

add_subdirectory(3rdparty)
add_subdirectory(src)