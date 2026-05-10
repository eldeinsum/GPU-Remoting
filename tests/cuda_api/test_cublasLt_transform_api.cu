#include <cublasLt.h>
#include <cuda_runtime.h>

#include <math.h>
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

static void check_close(float actual, float expected, const char *expr, const char *file, int line) {
    if (fabsf(actual - expected) > 1.0e-5f) {
        fprintf(stderr, "check failed: %s got %.7g expected %.7g at %s:%d\n", expr, actual, expected, file, line);
        exit(1);
    }
}

#define CHECK_CLOSE(actual, expected) check_close((actual), (expected), #actual, __FILE__, __LINE__)

static void test_matrix_transform_desc_attributes() {
    cublasLtMatrixTransformDesc_t desc = NULL;
    CHECK_CUBLAS(cublasLtMatrixTransformDescCreate(&desc, CUDA_R_32F));

    size_t written = 0;
    cudaDataType_t scale_type = CUDA_R_64F;
    CHECK_CUBLAS(cublasLtMatrixTransformDescGetAttribute(
        desc,
        CUBLASLT_MATRIX_TRANSFORM_DESC_SCALE_TYPE,
        &scale_type,
        sizeof(scale_type),
        &written));
    CHECK_TRUE(scale_type == CUDA_R_32F);
    CHECK_TRUE(written == sizeof(scale_type));

    cublasLtPointerMode_t pointer_mode = CUBLASLT_POINTER_MODE_DEVICE;
    written = 0;
    CHECK_CUBLAS(cublasLtMatrixTransformDescGetAttribute(
        desc,
        CUBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE,
        &pointer_mode,
        sizeof(pointer_mode),
        &written));
    CHECK_TRUE(pointer_mode == CUBLASLT_POINTER_MODE_HOST);
    CHECK_TRUE(written == sizeof(pointer_mode));

    pointer_mode = CUBLASLT_POINTER_MODE_DEVICE;
    CHECK_CUBLAS(cublasLtMatrixTransformDescSetAttribute(
        desc,
        CUBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE,
        &pointer_mode,
        sizeof(pointer_mode)));

    pointer_mode = CUBLASLT_POINTER_MODE_HOST;
    written = 0;
    CHECK_CUBLAS(cublasLtMatrixTransformDescGetAttribute(
        desc,
        CUBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE,
        &pointer_mode,
        sizeof(pointer_mode),
        &written));
    CHECK_TRUE(pointer_mode == CUBLASLT_POINTER_MODE_DEVICE);
    CHECK_TRUE(written == sizeof(pointer_mode));

    cublasOperation_t transa = CUBLAS_OP_T;
    CHECK_CUBLAS(cublasLtMatrixTransformDescSetAttribute(
        desc,
        CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA,
        &transa,
        sizeof(transa)));

    transa = CUBLAS_OP_N;
    written = 0;
    CHECK_CUBLAS(cublasLtMatrixTransformDescGetAttribute(
        desc,
        CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA,
        &transa,
        sizeof(transa),
        &written));
    CHECK_TRUE(transa == CUBLAS_OP_T);
    CHECK_TRUE(written == sizeof(transa));

    cublasOperation_t transb = CUBLAS_OP_N;
    CHECK_CUBLAS(cublasLtMatrixTransformDescSetAttribute(
        desc,
        CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSB,
        &transb,
        sizeof(transb)));

    transb = CUBLAS_OP_T;
    written = 0;
    CHECK_CUBLAS(cublasLtMatrixTransformDescGetAttribute(
        desc,
        CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSB,
        &transb,
        sizeof(transb),
        &written));
    CHECK_TRUE(transb == CUBLAS_OP_N);
    CHECK_TRUE(written == sizeof(transb));

    CHECK_CUBLAS(cublasLtMatrixTransformDescDestroy(desc));
}

static void check_output(const float *out, const float *a, const float *b, float alpha, float beta, int count) {
    for (int i = 0; i < count; ++i) {
        CHECK_CLOSE(out[i], alpha * a[i] + beta * b[i]);
    }
}

static void test_matrix_transform_operation() {
    const uint64_t rows = 2;
    const uint64_t cols = 3;
    const int count = rows * cols;
    const float host_a[count] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    const float host_b[count] = {6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
    float host_c[count] = {0.0f};

    float *dev_a = NULL;
    float *dev_b = NULL;
    float *dev_c = NULL;
    float *dev_alpha = NULL;
    float *dev_beta = NULL;

    CHECK_CUDA(cudaMalloc(&dev_a, sizeof(host_a)));
    CHECK_CUDA(cudaMalloc(&dev_b, sizeof(host_b)));
    CHECK_CUDA(cudaMalloc(&dev_c, sizeof(host_c)));
    CHECK_CUDA(cudaMalloc(&dev_alpha, sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dev_beta, sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dev_a, host_a, sizeof(host_a), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dev_b, host_b, sizeof(host_b), cudaMemcpyHostToDevice));

    cublasLtHandle_t handle = NULL;
    cublasLtMatrixLayout_t a_desc = NULL;
    cublasLtMatrixLayout_t b_desc = NULL;
    cublasLtMatrixLayout_t c_desc = NULL;
    cublasLtMatrixTransformDesc_t transform_desc = NULL;

    CHECK_CUBLAS(cublasLtCreate(&handle));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_32F, rows, cols, rows));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_32F, rows, cols, rows));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_32F, rows, cols, rows));
    CHECK_CUBLAS(cublasLtMatrixTransformDescCreate(&transform_desc, CUDA_R_32F));

    float alpha = 2.0f;
    float beta = 3.0f;
    CHECK_CUDA(cudaMemset(dev_c, 0, sizeof(host_c)));
    CHECK_CUBLAS(cublasLtMatrixTransform(
        handle,
        transform_desc,
        &alpha,
        dev_a,
        a_desc,
        &beta,
        dev_b,
        b_desc,
        dev_c,
        c_desc,
        0));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(host_c, dev_c, sizeof(host_c), cudaMemcpyDeviceToHost));
    check_output(host_c, host_a, host_b, alpha, beta, count);

    cublasLtPointerMode_t pointer_mode = CUBLASLT_POINTER_MODE_DEVICE;
    CHECK_CUBLAS(cublasLtMatrixTransformDescSetAttribute(
        transform_desc,
        CUBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE,
        &pointer_mode,
        sizeof(pointer_mode)));

    alpha = 1.0f;
    beta = -1.0f;
    CHECK_CUDA(cudaMemcpy(dev_alpha, &alpha, sizeof(alpha), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dev_beta, &beta, sizeof(beta), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dev_c, 0, sizeof(host_c)));
    CHECK_CUBLAS(cublasLtMatrixTransform(
        handle,
        transform_desc,
        dev_alpha,
        dev_a,
        a_desc,
        dev_beta,
        dev_b,
        b_desc,
        dev_c,
        c_desc,
        0));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(host_c, dev_c, sizeof(host_c), cudaMemcpyDeviceToHost));
    check_output(host_c, host_a, host_b, alpha, beta, count);

    CHECK_CUBLAS(cublasLtMatrixTransformDescDestroy(transform_desc));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(c_desc));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(b_desc));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(a_desc));
    CHECK_CUBLAS(cublasLtDestroy(handle));
    CHECK_CUDA(cudaFree(dev_beta));
    CHECK_CUDA(cudaFree(dev_alpha));
    CHECK_CUDA(cudaFree(dev_c));
    CHECK_CUDA(cudaFree(dev_b));
    CHECK_CUDA(cudaFree(dev_a));
}

int main() {
    test_matrix_transform_desc_attributes();
    test_matrix_transform_operation();
    printf("cuBLASLt matrix transform descriptor API test passed\n");
    return 0;
}
