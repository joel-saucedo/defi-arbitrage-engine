#!/bin/bash
# build script for c++ networking module

set -e

echo "[build] compiling c++ networking module with maximum optimizations..."

# create build directory
mkdir -p build
cd build

# configure with maximum optimization flags
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="-O3 -march=native -mtune=native -flto -ffast-math" \
  -DCMAKE_CXX_STANDARD=20

# build with all available cores
make -j$(nproc)

echo "[build] c++ module compiled successfully"
