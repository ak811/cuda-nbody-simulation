#!/bin/bash
set -e

# Create output folders
mkdir -p outputs/seq outputs/omp outputs/cuda outputs/vis outputs/plots

# Build sequential
g++ -std=c++17 -O3 nbody_seq.cpp -o nbody_seq

# Build OpenMP
g++ -std=c++17 -O3 -fopenmp nbody_omp.cpp -o nbody_omp

# Build CUDA
nvcc -std=c++17 -O3 nbody_cuda.cu -o nbody_cuda

echo "Build completed."
echo "Executables: ./nbody_seq ./nbody_omp ./nbody_cuda"
