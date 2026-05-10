#include <cublasLt.h>
#include <cuda_runtime.h>

#include <stdio.h>
#include <stdlib.h>

static void check_cublas(cublasStatus_t status, const char *expr, const char *file, int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLASLt error %d for %s at %s:%d\n", status, expr, file, line);
        exit(1);
    }
}

#define CHECK_CUBLAS(expr) check_cublas((expr), #expr, __FILE__, __LINE__)

static void check_true(bool condition, const char *expr, const char *file, int line) {
    if (!condition) {
        fprintf(stderr, "check failed: %s at %s:%d\n", expr, file, line);
        exit(1);
    }
}

#define CHECK_TRUE(expr) check_true((expr), #expr, __FILE__, __LINE__)

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

int main() {
    test_matrix_transform_desc_attributes();
    printf("cuBLASLt matrix transform descriptor API test passed\n");
    return 0;
}
