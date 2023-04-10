#pragma once

namespace ai::ml::neural
{
template<typename T>
class CudaVector
{
public:
    CudaVector(size_t size, T* inputData);
    ~CudaVector();

    void cudaConfigureKernelCall(size_t threadsPerBlock);

    void cudaKernelCall(double scalar);

    void cudaShowKernelCallResults();

private:
    size_t size;
    T* hostData;
    T* deviceData;
    size_t threadsPerBlock;
    size_t numBlocks;
};
} // namespace ai::ml::neural
