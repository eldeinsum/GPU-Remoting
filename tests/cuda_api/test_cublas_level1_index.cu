#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

template <typename T>
static T *to_device(const std::vector<T> &host) {
    T *device = nullptr;
    CHECK_CUDA(cudaMalloc(&device, host.size() * sizeof(T)));
    CHECK_CUDA(cudaMemcpy(device, host.data(), host.size() * sizeof(T),
                          cudaMemcpyHostToDevice));
    return device;
}

static void expect_index(int64_t got, int64_t want, const char *label) {
    if (got != want) {
        std::fprintf(stderr, "%s index mismatch: got %lld want %lld\n", label,
                     static_cast<long long>(got), static_cast<long long>(want));
        std::exit(1);
    }
}

template <typename T>
static cudaDataType_t cuda_type();

template <>
cudaDataType_t cuda_type<float>() {
    return CUDA_R_32F;
}

template <>
cudaDataType_t cuda_type<double>() {
    return CUDA_R_64F;
}

template <>
cudaDataType_t cuda_type<cuComplex>() {
    return CUDA_C_32F;
}

template <>
cudaDataType_t cuda_type<cuDoubleComplex>() {
    return CUDA_C_64F;
}

template <typename VecT, typename Fn, typename Fn64>
static void run_index_case(cublasHandle_t handle, const std::vector<VecT> &input,
                           int expected, Fn fn, Fn64 fn64,
                           const char *label) {
    const int n = static_cast<int>(input.size());
    VecT *device_input = to_device(input);

    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));

    int host_result = 0;
    CHECK_CUBLAS(fn(handle, n, device_input, 1, &host_result));
    expect_index(host_result, expected, label);

    int64_t host_result64 = 0;
    CHECK_CUBLAS(
        fn64(handle, static_cast<int64_t>(n), device_input, 1, &host_result64));
    expect_index(host_result64, expected, label);

    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));

    int zero = 0;
    int *device_result = nullptr;
    CHECK_CUDA(cudaMalloc(&device_result, sizeof(int)));
    CHECK_CUDA(cudaMemcpy(device_result, &zero, sizeof(int),
                          cudaMemcpyHostToDevice));
    CHECK_CUBLAS(fn(handle, n, device_input, 1, device_result));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(&host_result, device_result, sizeof(int),
                          cudaMemcpyDeviceToHost));
    expect_index(host_result, expected, label);

    int64_t zero64 = 0;
    int64_t *device_result64 = nullptr;
    CHECK_CUDA(cudaMalloc(&device_result64, sizeof(int64_t)));
    CHECK_CUDA(cudaMemcpy(device_result64, &zero64, sizeof(int64_t),
                          cudaMemcpyHostToDevice));
    CHECK_CUBLAS(fn64(handle, static_cast<int64_t>(n), device_input, 1,
                     device_result64));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(&host_result64, device_result64, sizeof(int64_t),
                          cudaMemcpyDeviceToHost));
    expect_index(host_result64, expected, label);

    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
    CHECK_CUDA(cudaFree(device_result64));
    CHECK_CUDA(cudaFree(device_result));
    CHECK_CUDA(cudaFree(device_input));
}

template <typename VecT, typename Fn, typename Fn64>
static void run_index_ex_case(cublasHandle_t handle,
                              const std::vector<VecT> &input, int expected,
                              Fn fn, Fn64 fn64, const char *label) {
    const int n = static_cast<int>(input.size());
    VecT *device_input = to_device(input);

    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));

    int host_result = 0;
    CHECK_CUBLAS(
        fn(handle, n, device_input, cuda_type<VecT>(), 1, &host_result));
    expect_index(host_result, expected, label);

    int64_t host_result64 = 0;
    CHECK_CUBLAS(fn64(handle, static_cast<int64_t>(n), device_input,
                     cuda_type<VecT>(), 1, &host_result64));
    expect_index(host_result64, expected, label);

    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));

    int zero = 0;
    int *device_result = nullptr;
    CHECK_CUDA(cudaMalloc(&device_result, sizeof(int)));
    CHECK_CUDA(cudaMemcpy(device_result, &zero, sizeof(int),
                          cudaMemcpyHostToDevice));
    CHECK_CUBLAS(
        fn(handle, n, device_input, cuda_type<VecT>(), 1, device_result));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(&host_result, device_result, sizeof(int),
                          cudaMemcpyDeviceToHost));
    expect_index(host_result, expected, label);

    int64_t zero64 = 0;
    int64_t *device_result64 = nullptr;
    CHECK_CUDA(cudaMalloc(&device_result64, sizeof(int64_t)));
    CHECK_CUDA(cudaMemcpy(device_result64, &zero64, sizeof(int64_t),
                          cudaMemcpyHostToDevice));
    CHECK_CUBLAS(fn64(handle, static_cast<int64_t>(n), device_input,
                     cuda_type<VecT>(), 1, device_result64));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(&host_result64, device_result64, sizeof(int64_t),
                          cudaMemcpyDeviceToHost));
    expect_index(host_result64, expected, label);

    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
    CHECK_CUDA(cudaFree(device_result64));
    CHECK_CUDA(cudaFree(device_result));
    CHECK_CUDA(cudaFree(device_input));
}

static void test_iamax(cublasHandle_t handle) {
    run_index_case<float>(
        handle, std::vector<float>{-1.0f, 4.0f, -2.0f}, 2,
        cublasIsamax_v2, cublasIsamax_v2_64, "Isamax");
    run_index_case<double>(
        handle, std::vector<double>{-1.0, 4.0, -2.0}, 2, cublasIdamax_v2,
        cublasIdamax_v2_64, "Idamax");
    run_index_case<cuComplex>(
        handle,
        std::vector<cuComplex>{{1.0f, 1.0f}, {-4.0f, 0.5f},
                               {0.25f, -0.25f}},
        2, cublasIcamax_v2, cublasIcamax_v2_64, "Icamax");
    run_index_case<cuDoubleComplex>(
        handle,
        std::vector<cuDoubleComplex>{{1.0, 1.0}, {-4.0, 0.5},
                                     {0.25, -0.25}},
        2, cublasIzamax_v2, cublasIzamax_v2_64, "Izamax");
    run_index_ex_case<float>(
        handle, std::vector<float>{-1.0f, 4.0f, -2.0f}, 2, cublasIamaxEx,
        cublasIamaxEx_64, "IamaxEx float");
    run_index_ex_case<double>(
        handle, std::vector<double>{-1.0, 4.0, -2.0}, 2, cublasIamaxEx,
        cublasIamaxEx_64, "IamaxEx double");
    run_index_ex_case<cuComplex>(
        handle,
        std::vector<cuComplex>{{1.0f, 1.0f}, {-4.0f, 0.5f},
                               {0.25f, -0.25f}},
        2, cublasIamaxEx, cublasIamaxEx_64, "IamaxEx complex-float");
    run_index_ex_case<cuDoubleComplex>(
        handle,
        std::vector<cuDoubleComplex>{{1.0, 1.0}, {-4.0, 0.5},
                                     {0.25, -0.25}},
        2, cublasIamaxEx, cublasIamaxEx_64, "IamaxEx complex-double");
}

static void test_iamin(cublasHandle_t handle) {
    run_index_case<float>(
        handle, std::vector<float>{-1.0f, 4.0f, -2.0f}, 1,
        cublasIsamin_v2, cublasIsamin_v2_64, "Isamin");
    run_index_case<double>(
        handle, std::vector<double>{-1.0, 4.0, -2.0}, 1, cublasIdamin_v2,
        cublasIdamin_v2_64, "Idamin");
    run_index_case<cuComplex>(
        handle,
        std::vector<cuComplex>{{1.0f, 1.0f}, {-4.0f, 0.5f},
                               {0.25f, -0.25f}},
        3, cublasIcamin_v2, cublasIcamin_v2_64, "Icamin");
    run_index_case<cuDoubleComplex>(
        handle,
        std::vector<cuDoubleComplex>{{1.0, 1.0}, {-4.0, 0.5},
                                     {0.25, -0.25}},
        3, cublasIzamin_v2, cublasIzamin_v2_64, "Izamin");
    run_index_ex_case<float>(
        handle, std::vector<float>{-1.0f, 4.0f, -2.0f}, 1, cublasIaminEx,
        cublasIaminEx_64, "IaminEx float");
    run_index_ex_case<double>(
        handle, std::vector<double>{-1.0, 4.0, -2.0}, 1, cublasIaminEx,
        cublasIaminEx_64, "IaminEx double");
    run_index_ex_case<cuComplex>(
        handle,
        std::vector<cuComplex>{{1.0f, 1.0f}, {-4.0f, 0.5f},
                               {0.25f, -0.25f}},
        3, cublasIaminEx, cublasIaminEx_64, "IaminEx complex-float");
    run_index_ex_case<cuDoubleComplex>(
        handle,
        std::vector<cuDoubleComplex>{{1.0, 1.0}, {-4.0, 0.5},
                                     {0.25, -0.25}},
        3, cublasIaminEx, cublasIaminEx_64, "IaminEx complex-double");
}

int main() {
    cublasHandle_t handle = nullptr;
    CHECK_CUBLAS(cublasCreate(&handle));
    test_iamax(handle);
    test_iamin(handle);
    CHECK_CUBLAS(cublasDestroy(handle));
    std::puts("cuBLAS Level-1 index test passed");
    return 0;
}
