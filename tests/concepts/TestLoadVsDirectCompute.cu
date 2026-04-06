#include <CppUTest/CommandLineTestRunner.h>
#include <CppUTest/TestHarness.h>

#include "LoadVsDirectCompute.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace
{
void checkCuda(cudaError_t err, const char* what)
{
    if (err != cudaSuccess)
    {
        printf("CUDA error %s: %s\n", what, cudaGetErrorString(err));
        std::abort();
    }
}

float hostReferenceTangentSum(float angle, unsigned workRepeats)
{
    float t = std::tan(static_cast<double>(angle));
    return static_cast<float>(static_cast<double>(workRepeats) * static_cast<double>(t));
}

void fillReplicatedLUT(const std::vector<float>& angles, unsigned workRepeats,
                       std::vector<float>& sin_lut, std::vector<float>& cos_lut)
{
    std::size_t count = angles.size();
    sin_lut.resize(count * static_cast<std::size_t>(workRepeats));
    cos_lut.resize(count * static_cast<std::size_t>(workRepeats));
    for (unsigned r = 0; r < workRepeats; ++r)
    {
        for (std::size_t i = 0; i < count; ++i)
        {
            std::size_t li = static_cast<std::size_t>(r) * count + i;
            sin_lut[li]    = std::sin(angles[i]);
            cos_lut[li]    = std::cos(angles[i]);
        }
    }
}

float millisecondsDirect(const float* d_angles, float* d_out, std::size_t count,
                         unsigned workRepeats, int threads, cudaEvent_t start, cudaEvent_t stop)
{
    int blocks = static_cast<int>((count + static_cast<std::size_t>(threads) - 1U) /
                                  static_cast<std::size_t>(threads));
    checkCuda(cudaEventRecord(start), "EventRecord start");
    gpu::directTangentCompute<<<blocks, threads>>>(d_angles, d_out, count, workRepeats);
    checkCuda(cudaGetLastError(), "kernel");
    checkCuda(cudaEventRecord(stop), "EventRecord stop");
    checkCuda(cudaEventSynchronize(stop), "EventSync");
    float ms = 0.f;
    checkCuda(cudaEventElapsedTime(&ms, start, stop), "ElapsedTime");
    return ms;
}

float millisecondsLUT(const float* d_sin, const float* d_cos, float* d_out, std::size_t count,
                      unsigned workRepeats, int threads, cudaEvent_t start, cudaEvent_t stop)
{
    int blocks = static_cast<int>((count + static_cast<std::size_t>(threads) - 1U) /
                                  static_cast<std::size_t>(threads));
    checkCuda(cudaEventRecord(start), "EventRecord start LUT");
    gpu::computeTangentFromLUT<<<blocks, threads>>>(d_sin, d_cos, d_out, count, workRepeats);
    checkCuda(cudaGetLastError(), "LUT kernel");
    checkCuda(cudaEventRecord(stop), "EventRecord stop LUT");
    checkCuda(cudaEventSynchronize(stop), "EventSync LUT");
    float ms = 0.f;
    checkCuda(cudaEventElapsedTime(&ms, start, stop), "ElapsedTime LUT");
    return ms;
}
}  // namespace

TEST_GROUP(LoadVsDirectCompute){void setup(){}

                                void teardown(){}};

TEST(LoadVsDirectCompute, ResultsMatchBetweenPaths)
{
    constexpr std::size_t count       = 4096;
    constexpr unsigned    workRepeats = 16;
    constexpr int         threads     = 256;

    std::vector<float> angles(count);
    for (std::size_t i = 0; i < count; ++i)
    {
        angles[i] = 0.1f + 0.017f * static_cast<float>(i % 500);
    }

    std::vector<float> sin_lut;
    std::vector<float> cos_lut;
    fillReplicatedLUT(angles, workRepeats, sin_lut, cos_lut);

    float *d_angles = nullptr, *d_out_direct = nullptr, *d_out_lut = nullptr;
    float *d_sin = nullptr, *d_cos = nullptr;
    checkCuda(cudaMalloc(&d_angles, count * sizeof(float)), "cudaMalloc angles");
    checkCuda(cudaMalloc(&d_out_direct, count * sizeof(float)), "cudaMalloc out direct");
    checkCuda(cudaMalloc(&d_out_lut, count * sizeof(float)), "cudaMalloc out lut");
    checkCuda(cudaMalloc(&d_sin, sin_lut.size() * sizeof(float)), "cudaMalloc sin");
    checkCuda(cudaMalloc(&d_cos, cos_lut.size() * sizeof(float)), "cudaMalloc cos");

    checkCuda(cudaMemcpy(d_angles, angles.data(), count * sizeof(float), cudaMemcpyHostToDevice),
              "copy angles");
    checkCuda(
        cudaMemcpy(d_sin, sin_lut.data(), sin_lut.size() * sizeof(float), cudaMemcpyHostToDevice),
        "copy sin");
    checkCuda(
        cudaMemcpy(d_cos, cos_lut.data(), cos_lut.size() * sizeof(float), cudaMemcpyHostToDevice),
        "copy cos");

    int blocks = static_cast<int>((count + static_cast<std::size_t>(threads) - 1U) /
                                  static_cast<std::size_t>(threads));
    gpu::directTangentCompute<<<blocks, threads>>>(d_angles, d_out_direct, count, workRepeats);
    checkCuda(cudaGetLastError(), "direct");
    gpu::computeTangentFromLUT<<<blocks, threads>>>(d_sin, d_cos, d_out_lut, count, workRepeats);
    checkCuda(cudaGetLastError(), "lut");

    std::vector<float> host_direct(count);
    std::vector<float> host_lut(count);
    checkCuda(
        cudaMemcpy(host_direct.data(), d_out_direct, count * sizeof(float), cudaMemcpyDeviceToHost),
        "dtoh direct");
    checkCuda(cudaMemcpy(host_lut.data(), d_out_lut, count * sizeof(float), cudaMemcpyDeviceToHost),
              "dtoh lut");

    checkCuda(cudaFree(d_angles), "free");
    checkCuda(cudaFree(d_out_direct), "free");
    checkCuda(cudaFree(d_out_lut), "free");
    checkCuda(cudaFree(d_sin), "free");
    checkCuda(cudaFree(d_cos), "free");

    for (std::size_t i = 0; i < count; ++i)
    {
        float        ref = hostReferenceTangentSum(angles[i], workRepeats);
        const double tol = std::max(1e-2, 1e-5 * std::fabs(static_cast<double>(ref)));
        DOUBLES_EQUAL(ref, host_direct[i], tol);
        DOUBLES_EQUAL(ref, host_lut[i], tol);
        DOUBLES_EQUAL(host_direct[i], host_lut[i], tol);
    }
}

TEST(LoadVsDirectCompute, TippingPointSweep)
{
    // Sized so peak LUT (sin+cos) stays near ~1 GiB: count * maxRepeats * 2 * sizeof(float).
    constexpr std::size_t count      = 1u << 20;
    constexpr int         threads    = 256;
    constexpr int         warmupRuns = 3;
    constexpr int         timedRuns  = 12;

    std::vector<float> angles(count);
    for (std::size_t i = 0; i < count; ++i)
    {
        angles[i] = -0.45f + 0.0003f * static_cast<float>(i & 0xffffu);
    }

    float *d_angles = nullptr, *d_out = nullptr, *d_sin = nullptr, *d_cos = nullptr;
    checkCuda(cudaMalloc(&d_angles, count * sizeof(float)), "cudaMalloc angles sweep");
    checkCuda(cudaMalloc(&d_out, count * sizeof(float)), "cudaMalloc out sweep");
    checkCuda(cudaMemcpy(d_angles, angles.data(), count * sizeof(float), cudaMemcpyHostToDevice),
              "copy angles sweep");

    cudaEvent_t start{};
    cudaEvent_t stop{};
    checkCuda(cudaEventCreate(&start), "event create");
    checkCuda(cudaEventCreate(&stop), "event create 2");

    printf("\nLoad vs on-the-fly compute (same math per repeat; LUT uses replicated planes).\n");
    printf("count=%zu threads=%d timedRuns=%d (ms averages).\n", count, threads, timedRuns);
    printf("%8s %12s %12s %10s\n", "repeats", "direct_ms", "lut_ms", "lut/direct");

    const unsigned repeatValues[] = {1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128};

    for (unsigned workRepeats : repeatValues)
    {
        std::vector<float> sin_lut;
        std::vector<float> cos_lut;
        fillReplicatedLUT(angles, workRepeats, sin_lut, cos_lut);

        if (d_sin != nullptr)
        {
            checkCuda(cudaFree(d_sin), "free sin prev");
        }
        if (d_cos != nullptr)
        {
            checkCuda(cudaFree(d_cos), "free cos prev");
        }
        checkCuda(cudaMalloc(&d_sin, sin_lut.size() * sizeof(float)), "cudaMalloc sin sweep");
        checkCuda(cudaMalloc(&d_cos, cos_lut.size() * sizeof(float)), "cudaMalloc cos sweep");
        checkCuda(cudaMemcpy(d_sin, sin_lut.data(), sin_lut.size() * sizeof(float),
                             cudaMemcpyHostToDevice),
                  "copy sin sweep");
        checkCuda(cudaMemcpy(d_cos, cos_lut.data(), cos_lut.size() * sizeof(float),
                             cudaMemcpyHostToDevice),
                  "copy cos sweep");

        int blocks = static_cast<int>((count + static_cast<std::size_t>(threads) - 1U) /
                                      static_cast<std::size_t>(threads));

        for (int w = 0; w < warmupRuns; ++w)
        {
            gpu::directTangentCompute<<<blocks, threads>>>(d_angles, d_out, count, workRepeats);
            gpu::computeTangentFromLUT<<<blocks, threads>>>(d_sin, d_cos, d_out, count,
                                                            workRepeats);
        }
        checkCuda(cudaDeviceSynchronize(), "sync warmup");

        float sumDirect = 0.f;
        float sumLut    = 0.f;
        for (int t = 0; t < timedRuns; ++t)
        {
            sumDirect +=
                millisecondsDirect(d_angles, d_out, count, workRepeats, threads, start, stop);
            sumLut +=
                millisecondsLUT(d_sin, d_cos, d_out, count, workRepeats, threads, start, stop);
        }

        float msDirect = sumDirect / static_cast<float>(timedRuns);
        float msLut    = sumLut / static_cast<float>(timedRuns);
        float ratio    = msLut / msDirect;
        printf("%8u %12.4f %12.4f %10.3f\n", workRepeats, msDirect, msLut, ratio);
    }

    checkCuda(cudaEventDestroy(start), "destroy start");
    checkCuda(cudaEventDestroy(stop), "destroy stop");
    checkCuda(cudaFree(d_angles), "free angles sweep");
    checkCuda(cudaFree(d_out), "free out sweep");
    checkCuda(cudaFree(d_sin), "free sin sweep");
    checkCuda(cudaFree(d_cos), "free cos sweep");
}

int main(int argc, char** argv)
{
    return CommandLineTestRunner::RunAllTests(argc, argv);
}
