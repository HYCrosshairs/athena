#include "CudaVector.cuh"

#include <stdexcept>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

using namespace ai::ml::neural;

template<typename T>
__global__ void vectorMultiplyBy(T* dataInput, double scalar, size_t size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size)
    {
        dataInput[tid] = dataInput[tid] * scalar;
    }       
}

template<typename T>
CudaVector<T>::CudaVector(size_t size, T* inputData) : size(size), hostData(inputData)
{
    cudaMalloc((void **)&deviceData, size * sizeof(T));

    cudaMemcpy(deviceData, hostData, size * sizeof(T), cudaMemcpyHostToDevice);
}

template<typename T>
CudaVector<T>::~CudaVector()
{
    cudaFree(deviceData);
}

template<typename T>
void CudaVector<T>::cudaConfigureKernelCall(size_t threadsPerBlock)
{
    this->threadsPerBlock = threadsPerBlock;
    this->numBlocks = (this->size + threadsPerBlock - 1) / threadsPerBlock;
}

template<typename T>
void CudaVector<T>::cudaKernelCall(double scalar)
{
    vectorMultiplyBy<<<this->numBlocks, this->threadsPerBlock>>>(deviceData, scalar, this->size);

    cudaError_t error = cudaGetLastError();
    
    if (error != cudaSuccess)
    {
        throw std::runtime_error(cudaGetErrorString(error));
    }

    cudaMemcpy(hostData, deviceData, size * sizeof(T), cudaMemcpyDeviceToHost);
}

template<typename T>
void CudaVector<T>::cudaShowKernelCallResults()
{
    for (size_t i = 0; i < this->size; i++)
    {
        std::cout << hostData[i] << std::endl;
    }
    
}

template class CudaVector<double>;