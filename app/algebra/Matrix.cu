#include "SystemDefinitions.hpp"

#include <cuda_stdint.h>

// CUDA error checking macro
#define CUDA_CHECK(call)\
{\
cudaError_t err = call;\
if (err != cudaSuccess)\
{\
std::cerr << "CUDA error in " << #call << " at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl;\
exit(1);\
}\
}

template<typename T, uint32_t rows, uint32_t cols>
class Matrix<T, rows, cols>
{
public:
    // Constructor - allocates device memory
    Matrix() {
        CUDA_CHECK(cudaMalloc(&d_matrix, rows * cols * sizeof(T)));
        CUDA_CHECK(cudaMemset(d_matrix, 0, rows * cols * sizeof(T)));
    }

    // Copy constructor
    Matrix(const Matrix& other) {
        CUDA_CHECK(cudaMalloc(&d_matrix, rows * cols * sizeof(T)));
        CUDA_CHECK(cudaMemcpy(d_matrix, other.d_matrix, rows * cols * sizeof(T), cudaMemcpyDeviceToDevice));
        syncHostFromDevice(); // Update host copy
    }

    // Destructor - frees device memory
    ~Matrix() {
        if (d_matrix) {
            cudaFree(d_matrix);
        }
    }

    // Copy assignment operator
    Matrix& operator=(const Matrix& other) {
        if (this != &other) {
            CUDA_CHECK(cudaMemcpy(d_matrix, other.d_matrix, rows * cols * sizeof(T), cudaMemcpyDeviceToDevice));
            syncHostFromDevice(); // Update host copy
        }
        return *this;
    }

    // Synchronize host memory from device
    void syncHostFromDevice() {
        CUDA_CHECK(cudaMemcpy(&h_matrix[0][0], d_matrix, rows * cols * sizeof(T), cudaMemcpyDeviceToHost));
    }

    // Synchronize device memory from host
    void syncDeviceFromHost() {
        CUDA_CHECK(cudaMemcpy(d_matrix, &h_matrix[0][0], rows * cols * sizeof(T), cudaMemcpyHostToDevice));
    }

    void print() {
        syncHostFromDevice(); // Ensure host copy is up to date
        for (uint32_t rowIndex = 0; rowIndex < rows; rowIndex++) {
            for (uint32_t colIndex = 0; colIndex < cols; colIndex++) {
                std::cout << std::format("{:<8}", h_matrix[rowIndex][colIndex]);
            }
            std::cout << std::endl;
        }
    }

    void setValue(uint32_t row, uint32_t col, T value) {
        h_matrix[row][col] = value;
        // Update device memory immediately or user can call syncDeviceFromHost() later
        syncDeviceFromHost();
    }

    T getValue(uint32_t row, uint32_t col) const {
        // Note: This returns from host copy. For frequent access, call syncHostFromDevice() first
        return h_matrix[row][col];
    }

    // CUDA kernel for matrix addition
    __global__ static void addKernel(const T* a, const T* b, T* result, uint32_t n) {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            result[idx] = a[idx] + b[idx];
        }
    }

    // CUDA kernel for matrix multiplication
    __global__ static void multiplyKernel(const T* a, const T* b, T* result,
                                         uint32_t rowss, uint32_t colss, uint32_t common_dim) {
        uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
        uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row < rows && col < cols) {
            T sum = 0;
            for (uint32_t k = 0; k < common_dim; k++) {
                sum += a[row * common_dim + k] * b[k * cols + col];
            }
            result[row * cols + col] = sum;
        }
    }

    Matrix operator+(const Matrix& other) const noexcept {
        Matrix result;

        constexpr uint32_t N = rows * cols;
        const uint32_t blockSize = 256;
        const uint32_t numBlocks = (N + blockSize - 1) / blockSize;

        addKernel<<<numBlocks, blockSize>>>(d_matrix, other.d_matrix, result.d_matrix, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        result.syncHostFromDevice();
        return result;
    }

    Matrix operator*(const Matrix& other) const {
        static_assert(cols == rows, "Matrix size doesn't match");

        Matrix result;

        // Use 2D grid and blocks for matrix multiplication
        dim3 blockDim(16, 16);
        dim3 gridDim((cols + blockDim.x - 1) / blockDim.x,
                     (rows + blockDim.y - 1) / blockDim.y);

        multiplyKernel<<<gridDim, blockDim>>>(d_matrix, other.d_matrix, result.d_matrix,
                                             rows, cols, cols);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        result.syncHostFromDevice();
        return result;
    }

    // Get device pointer for direct kernel access
    T* deviceData() const { return d_matrix; }

    // Get host pointer
    T* hostData() { return &h_matrix[0][0]; }

    // Method to explicitly sync if user wants to batch host updates
    void syncToDevice() {
        syncDeviceFromHost();
    }

    void syncToHost() {
        syncHostFromDevice();
    }

private:
    T* d_matrix; // Device pointer
    T h_matrix[rows][cols]{}; // Host copy for printing/access
};