#include "Vectorization.cuh"

#include <iostream>
#include <vector>

__global__ void multiplyVectorBy(double* x, double scalar, size_t size)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
    {
        x[i] = x[i] * scalar;
    }
}

void function()
{
    const int N = 3;
    const int size = N * sizeof(double);

    double h_x[N] = {3, 5, 7};
    double* d_x;

    double scalar = 3;
    size_t threadsPerBlock = 256;
    size_t numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;

    cudaMalloc((void**)&d_x, size);

    // Initialize device array
    cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);

    // Launch kernel
    multiplyVectorBy<<<numBlocks, threadsPerBlock>>>(d_x, scalar, N);

    // Copy results from device to host
    cudaMemcpy(h_x, d_x, size, cudaMemcpyDeviceToHost);

    // Print results
    for (int i = 0; i < N; ++i)
    {
        std::cout << "h_x[" << i << "] = " << h_x[i] << std::endl;
    }

    // Free memory
    cudaFree(d_x);
}