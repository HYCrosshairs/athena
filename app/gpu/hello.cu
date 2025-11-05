#include <cuda_runtime.h>
#include <iostream>

__global__ void hello_kernel()
{
    printf("Hello from GPU!\n");
}

void gpu_hello()
{
    hello_kernel<<<1,1>>>();
    cudaDeviceSynchronize();
}
