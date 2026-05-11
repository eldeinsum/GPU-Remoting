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
        fprintf(
            stderr,
            "CUDA error %d (%s) for %s at %s:%d\n",
            status,
            cudaGetErrorString(status),
            expr,
            file,
            line);
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

template <typename T>
static T get_layout_attr(cublasLtMatrixLayout_t layout, cublasLtMatrixLayoutAttribute_t attr) {
    T value{};
    size_t written = 0;
    CHECK_CUBLAS(cublasLtMatrixLayoutGetAttribute(layout, attr, &value, sizeof(value), &written));
    CHECK_TRUE(written == sizeof(value));
    return value;
}

int main() {
    const int group_count = 3;
    const int64_t rows[] = {8, 16, 32};
    const int64_t cols[] = {4, 8, 16};
    const int64_t ld[] = {8, 16, 32};

    int64_t *d_rows = NULL;
    int64_t *d_cols = NULL;
    int64_t *d_ld = NULL;
    CHECK_CUDA(cudaMalloc(&d_rows, sizeof(rows)));
    CHECK_CUDA(cudaMalloc(&d_cols, sizeof(cols)));
    CHECK_CUDA(cudaMalloc(&d_ld, sizeof(ld)));
    CHECK_CUDA(cudaMemcpy(d_rows, rows, sizeof(rows), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_cols, cols, sizeof(cols), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_ld, ld, sizeof(ld), cudaMemcpyHostToDevice));

    cublasLtMatrixLayout_t layout = NULL;
    CHECK_CUBLAS(cublasLtGroupedMatrixLayoutCreate(
        &layout,
        CUDA_R_32F,
        group_count,
        d_rows,
        d_cols,
        d_ld));
    CHECK_TRUE(layout != NULL);

    CHECK_TRUE(get_layout_attr<cublasLtBatchMode_t>(layout, CUBLASLT_MATRIX_LAYOUT_BATCH_MODE) ==
               CUBLASLT_BATCH_MODE_GROUPED);
    CHECK_TRUE(
        get_layout_attr<int32_t>(layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT) == group_count);
    CHECK_TRUE(get_layout_attr<void *>(layout, CUBLASLT_GROUPED_MATRIX_LAYOUT_ROWS_ARRAY) == d_rows);
    CHECK_TRUE(get_layout_attr<void *>(layout, CUBLASLT_GROUPED_MATRIX_LAYOUT_COLS_ARRAY) == d_cols);
    CHECK_TRUE(get_layout_attr<void *>(layout, CUBLASLT_GROUPED_MATRIX_LAYOUT_LD_ARRAY) == d_ld);

    cublasLtIntegerWidth_t rows_cols_width = CUBLASLT_INTEGER_WIDTH_64;
    CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(
        layout,
        CUBLASLT_GROUPED_MATRIX_LAYOUT_ROWS_COLS_ARRAY_INTEGER_WIDTH,
        &rows_cols_width,
        sizeof(rows_cols_width)));
    CHECK_TRUE(get_layout_attr<cublasLtIntegerWidth_t>(
                   layout,
                   CUBLASLT_GROUPED_MATRIX_LAYOUT_ROWS_COLS_ARRAY_INTEGER_WIDTH) ==
               CUBLASLT_INTEGER_WIDTH_64);

    cublasLtIntegerWidth_t ld_width = CUBLASLT_INTEGER_WIDTH_64;
    CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(
        layout,
        CUBLASLT_GROUPED_MATRIX_LAYOUT_LD_ARRAY_INTEGER_WIDTH,
        &ld_width,
        sizeof(ld_width)));
    CHECK_TRUE(get_layout_attr<cublasLtIntegerWidth_t>(
                   layout,
                   CUBLASLT_GROUPED_MATRIX_LAYOUT_LD_ARRAY_INTEGER_WIDTH) ==
               CUBLASLT_INTEGER_WIDTH_64);

    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(layout));
    CHECK_CUDA(cudaFree(d_ld));
    CHECK_CUDA(cudaFree(d_cols));
    CHECK_CUDA(cudaFree(d_rows));

    printf("cuBLASLt grouped layout API test passed\n");
    return 0;
}
