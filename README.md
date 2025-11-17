# Athena Project

## Overview
Athena is a C++/CUDA project for learning GPU programming, deep learning, and 5G NR algorithms. It supports CPU/GPU builds, unit testing, benchmarking, and Matlab reference data generation.

## Structure
- `app/` - Core algorithms (algebra, nr, common)
- `cmake/` - CMake helper scripts and toolchains
- `ext/` - Third-party libraries (Google Benchmark, CppUTest, etc.)
- `matlab/` - Matlab scripts for reference/test data
- `sys/` - System definitions
- `test/` - Unit and performance tests

## Build Instructions
```sh
cmake -DTARGET_DEVICE=GPU -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain.cmake .
cmake --build .
```

## CPU/GPU Selection
Set `TARGET_DEVICE` to `CPU` or `GPU` to control build type.

## Adding Algorithms
- Place algebra code in `app/algebra/`
- Place 5G NR code in `app/nr/`
- Add Matlab scripts in `matlab/`

## License
MIT
