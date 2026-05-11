#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>

#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__,        \
                         __LINE__, cudaGetErrorString(err__));              \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

#define CHECK_CUBLAS(call)                                                   \
    do {                                                                     \
        cublasStatus_t err__ = (call);                                       \
        if (err__ != CUBLAS_STATUS_SUCCESS) {                                \
            std::fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__,      \
                         __LINE__, static_cast<int>(err__));                 \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

static bool close_value(float got, float want) {
    return std::fabs(got - want) <= 1e-4f;
}

static bool close_value(double got, double want) {
    return std::fabs(got - want) <= 1e-10;
}

template <typename T>
static T *to_device(const std::vector<T> &host) {
    T *device = nullptr;
    CHECK_CUDA(cudaMalloc(&device, host.size() * sizeof(T)));
    CHECK_CUDA(cudaMemcpy(device, host.data(), host.size() * sizeof(T),
                          cudaMemcpyHostToDevice));
    return device;
}

template <typename ResultT, typename VecT, typename Fn, typename Fn64>
static void run_reduction_case(cublasHandle_t handle,
                               const std::vector<VecT> &input,
                               ResultT expected, Fn fn, Fn64 fn64,
                               const char *label) {
    const int n = static_cast<int>(input.size());
    VecT *device_input = to_device(input);

    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));

    ResultT host_result = ResultT{};
    CHECK_CUBLAS(fn(handle, n, device_input, 1, &host_result));
    if (!close_value(host_result, expected)) {
        std::fprintf(stderr, "%s host result mismatch\n", label);
        std::exit(1);
    }

    host_result = ResultT{};
    CHECK_CUBLAS(
        fn64(handle, static_cast<int64_t>(n), device_input, 1, &host_result));
    if (!close_value(host_result, expected)) {
        std::fprintf(stderr, "%s host 64-bit result mismatch\n", label);
        std::exit(1);
    }

    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));

    ResultT zero = ResultT{};
    ResultT *device_result = nullptr;
    CHECK_CUDA(cudaMalloc(&device_result, sizeof(ResultT)));
    CHECK_CUDA(cudaMemcpy(device_result, &zero, sizeof(ResultT),
                          cudaMemcpyHostToDevice));

    CHECK_CUBLAS(fn(handle, n, device_input, 1, device_result));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(&host_result, device_result, sizeof(ResultT),
                          cudaMemcpyDeviceToHost));
    if (!close_value(host_result, expected)) {
        std::fprintf(stderr, "%s device result mismatch\n", label);
        std::exit(1);
    }

    CHECK_CUDA(cudaMemcpy(device_result, &zero, sizeof(ResultT),
                          cudaMemcpyHostToDevice));
    CHECK_CUBLAS(fn64(handle, static_cast<int64_t>(n), device_input, 1,
                     device_result));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(&host_result, device_result, sizeof(ResultT),
                          cudaMemcpyDeviceToHost));
    if (!close_value(host_result, expected)) {
        std::fprintf(stderr, "%s device 64-bit result mismatch\n", label);
        std::exit(1);
    }

    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
    CHECK_CUDA(cudaFree(device_result));
    CHECK_CUDA(cudaFree(device_input));
}

static void test_nrm2(cublasHandle_t handle) {
    run_reduction_case<float>(
        handle, std::vector<float>{3.0f, 4.0f, 12.0f}, 13.0f,
        cublasSnrm2_v2, cublasSnrm2_v2_64, "Snrm2");
    run_reduction_case<double>(
        handle, std::vector<double>{3.0, 4.0, 12.0}, 13.0,
        cublasDnrm2_v2, cublasDnrm2_v2_64, "Dnrm2");
    run_reduction_case<float>(
        handle, std::vector<cuComplex>{{3.0f, 4.0f}, {1.0f, -2.0f}},
        std::sqrt(30.0f), cublasScnrm2_v2, cublasScnrm2_v2_64, "Scnrm2");
    run_reduction_case<double>(
        handle, std::vector<cuDoubleComplex>{{3.0, 4.0}, {1.0, -2.0}},
        std::sqrt(30.0), cublasDznrm2_v2, cublasDznrm2_v2_64, "Dznrm2");
}

static void test_asum(cublasHandle_t handle) {
    run_reduction_case<float>(
        handle, std::vector<float>{-3.0f, 4.0f, -12.0f}, 19.0f,
        cublasSasum_v2, cublasSasum_v2_64, "Sasum");
    run_reduction_case<double>(
        handle, std::vector<double>{-3.0, 4.0, -12.0}, 19.0,
        cublasDasum_v2, cublasDasum_v2_64, "Dasum");
    run_reduction_case<float>(
        handle, std::vector<cuComplex>{{-3.0f, 4.0f}, {1.0f, -2.0f}},
        10.0f, cublasScasum_v2, cublasScasum_v2_64, "Scasum");
    run_reduction_case<double>(
        handle, std::vector<cuDoubleComplex>{{-3.0, 4.0}, {1.0, -2.0}},
        10.0, cublasDzasum_v2, cublasDzasum_v2_64, "Dzasum");
}

int main() {
    cublasHandle_t handle = nullptr;
    CHECK_CUBLAS(cublasCreate(&handle));
    test_nrm2(handle);
    test_asum(handle);
    CHECK_CUBLAS(cublasDestroy(handle));
    std::puts("cuBLAS Level-1 reduction test passed");
    return 0;
}
