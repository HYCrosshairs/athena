#include "Vectorization.cuh"

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void multiplyVectorBy(int* x, int scalar, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Thread %d is processing element %d\n", i, x[i]);
    if (i < size)
    {
        printf("Thread %d is processing element %d\n", i, x[i]);
        x[i] = x[i] * scalar;
    }
}

void Vectorization()
{
    const int N = 3;
    const int size = N * sizeof(int);

    int h_x[N];
    for (size_t i = 0; i < N; i++)
    {
        h_x[i] = 2 * i + 3;
    }
    
    int* d_x;

    int scalar = 3;
    size_t threadsPerBlock = 256;
    size_t numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    cudaMalloc(&d_x, size);

    // Initialize device array
    cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);

    // Launch kernel
    multiplyVectorBy<<<numBlocks, threadsPerBlock>>>(d_x, scalar, N);

    cudaDeviceSynchronize(); // Wait for the kernel to complete
    
    // Copy results from device to host
    cudaMemcpy(h_x, d_x, size, cudaMemcpyDeviceToHost);
    // Free memory
    cudaFree(d_x);

    // Print results
    for (int i = 0; i < N; ++i)
    {
        std::cout << "h_x[" << i << "] = " << h_x[i] << std::endl;
    }    
}
