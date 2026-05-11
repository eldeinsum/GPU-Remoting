#include <cublasLt.h>
#include <cuda_runtime.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static void check_cublas(cublasStatus_t status, const char *expr, const char *file, int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLASLt error %d for %s at %s:%d\n", status, expr, file, line);
        exit(1);
    }
}

#define CHECK_CUBLAS(expr) check_cublas((expr), #expr, __FILE__, __LINE__)

static void check_cuda(cudaError_t status, const char *expr, const char *file, int line) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error %d for %s at %s:%d\n", status, expr, file, line);
        exit(1);
    }
}

#define CHECK_CUDA(expr) check_cuda((expr), #expr, __FILE__, __LINE__)

static void check_true(bool condition, const char *expr, const char *file, int line) {
    if (!condition) {
        fprintf(stderr, "check failed: %s at %s:%d\n", expr, file, line);
        exit(1);
    }
}

#define CHECK_TRUE(expr) check_true((expr), #expr, __FILE__, __LINE__)

int main() {
    cublasLtEmulationDesc_t desc = NULL;
    CHECK_CUBLAS(cublasLtEmulationDescCreate(&desc));

    int value = -1;
    size_t written = 0;
    CHECK_CUBLAS(cublasLtEmulationDescGetAttribute(
        desc,
        CUBLASLT_EMULATION_DESC_STRATEGY,
        &value,
        sizeof(value),
        &written));
    CHECK_TRUE(value == 0);
    CHECK_TRUE(written == sizeof(value));

    value = 7;
    CHECK_CUBLAS(cublasLtEmulationDescSetAttribute(
        desc,
        CUBLASLT_EMULATION_DESC_FIXEDPOINT_MAX_MANTISSA_BIT_COUNT,
        &value,
        sizeof(value)));

    value = 0;
    written = 0;
    CHECK_CUBLAS(cublasLtEmulationDescGetAttribute(
        desc,
        CUBLASLT_EMULATION_DESC_FIXEDPOINT_MAX_MANTISSA_BIT_COUNT,
        &value,
        sizeof(value),
        &written));
    CHECK_TRUE(value == 7);
    CHECK_TRUE(written == sizeof(value));

    value = 2;
    CHECK_CUBLAS(cublasLtEmulationDescSetAttribute(
        desc,
        CUBLASLT_EMULATION_DESC_FIXEDPOINT_MANTISSA_BIT_OFFSET,
        &value,
        sizeof(value)));

    value = 0;
    written = 0;
    CHECK_CUBLAS(cublasLtEmulationDescGetAttribute(
        desc,
        CUBLASLT_EMULATION_DESC_FIXEDPOINT_MANTISSA_BIT_OFFSET,
        &value,
        sizeof(value),
        &written));
    CHECK_TRUE(value == 2);
    CHECK_TRUE(written == sizeof(value));

    int *device_count = NULL;
    int *round_trip = NULL;
    CHECK_CUDA(cudaMalloc(&device_count, sizeof(int)));
    CHECK_CUBLAS(cublasLtEmulationDescSetAttribute(
        desc,
        CUBLASLT_EMULATION_DESC_FIXEDPOINT_MANTISSA_BIT_COUNT_POINTER,
        &device_count,
        sizeof(device_count)));

    written = 0;
    CHECK_CUBLAS(cublasLtEmulationDescGetAttribute(
        desc,
        CUBLASLT_EMULATION_DESC_FIXEDPOINT_MANTISSA_BIT_COUNT_POINTER,
        &round_trip,
        sizeof(round_trip),
        &written));
    CHECK_TRUE(round_trip == device_count);
    CHECK_TRUE(written == sizeof(round_trip));

    CHECK_CUBLAS(cublasLtEmulationDescDestroy(desc));
    CHECK_CUDA(cudaFree(device_count));

    printf("cuBLASLt emulation descriptor API test passed\n");
    return 0;
}
