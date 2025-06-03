cmake_minimum_required(VERSION 3.16)
project(mev_network VERSION 1.0 LANGUAGES CXX)

# require c++20 for maximum optimization features
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# maximum optimization flags for high-frequency trading
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native -mavx2 -mfma -flto -ffast-math -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -mavx2 -mfma")

# find required packages
find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)

# check for boost libraries (header-only)
find_package(Boost REQUIRED COMPONENTS system)

# pybind11 for python integration
find_package(pybind11 REQUIRED)

# networking library target
add_library(mev_network SHARED
    src/network_monitor.cpp
)

# compiler optimizations for latency-critical code
target_compile_options(mev_network PRIVATE
    -O3
    -march=native
    -mtune=native
    -mavx2
    -mfma
    -flto
    -ffast-math
    -fno-rtti
    -finline-functions
    -funroll-loops
    -fprefetch-loop-arrays
)

# link libraries
target_link_libraries(mev_network
    Threads::Threads
    Boost::system
)

# python binding module
pybind11_add_module(mev_net_py src/python_bindings.cpp)
target_link_libraries(mev_net_py PRIVATE mev_network)

# set output directory
set_target_properties(mev_network mev_net_py
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

# install targets
install(TARGETS mev_network mev_net_py
    LIBRARY DESTINATION lib
)
