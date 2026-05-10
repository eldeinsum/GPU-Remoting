#include <cublasLt.h>
#include <cuda_runtime.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

static void test_property_queries() {
    int major = 0;
    int minor = 0;
    int patch = 0;

    CHECK_CUBLAS(cublasLtGetProperty(MAJOR_VERSION, &major));
    CHECK_CUBLAS(cublasLtGetProperty(MINOR_VERSION, &minor));
    CHECK_CUBLAS(cublasLtGetProperty(PATCH_LEVEL, &patch));

    CHECK_TRUE(major > 0);
    CHECK_TRUE(minor >= 0);
    CHECK_TRUE(patch >= 0);

    printf("cuBLASLt version property: %d.%d.%d\n", major, minor, patch);
}

static void test_version_and_status_helpers() {
    size_t version = cublasLtGetVersion();
    size_t cudart_version = cublasLtGetCudartVersion();
    CHECK_TRUE(version > 0);
    CHECK_TRUE(cudart_version > 0);

    const char *success_name = cublasLtGetStatusName(CUBLAS_STATUS_SUCCESS);
    const char *invalid_value_text = cublasLtGetStatusString(CUBLAS_STATUS_INVALID_VALUE);
    CHECK_TRUE(success_name != NULL);
    CHECK_TRUE(invalid_value_text != NULL);
    CHECK_TRUE(strstr(success_name, "SUCCESS") != NULL);
    CHECK_TRUE(strlen(invalid_value_text) > 0);
}

static void test_heuristics_cache_capacity() {
    size_t capacity = 0;
    CHECK_CUBLAS(cublasLtHeuristicsCacheGetCapacity(&capacity));
    CHECK_CUBLAS(cublasLtHeuristicsCacheSetCapacity(capacity));

    size_t round_trip = 0;
    CHECK_CUBLAS(cublasLtHeuristicsCacheGetCapacity(&round_trip));
    CHECK_TRUE(round_trip == capacity);
}

static void test_matmul_desc_attributes() {
    cublasLtMatmulDesc_t desc = NULL;
    CHECK_CUBLAS(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

    cublasLtPointerMode_t pointer_mode = CUBLASLT_POINTER_MODE_DEVICE;
    size_t written = 0;
    CHECK_CUBLAS(cublasLtMatmulDescGetAttribute(
        desc,
        CUBLASLT_MATMUL_DESC_POINTER_MODE,
        &pointer_mode,
        sizeof(pointer_mode),
        &written));
    CHECK_TRUE(pointer_mode == CUBLASLT_POINTER_MODE_HOST);
    CHECK_TRUE(written == sizeof(pointer_mode));

    pointer_mode = CUBLASLT_POINTER_MODE_DEVICE;
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        desc,
        CUBLASLT_MATMUL_DESC_POINTER_MODE,
        &pointer_mode,
        sizeof(pointer_mode)));

    pointer_mode = CUBLASLT_POINTER_MODE_HOST;
    written = 0;
    CHECK_CUBLAS(cublasLtMatmulDescGetAttribute(
        desc,
        CUBLASLT_MATMUL_DESC_POINTER_MODE,
        &pointer_mode,
        sizeof(pointer_mode),
        &written));
    CHECK_TRUE(pointer_mode == CUBLASLT_POINTER_MODE_DEVICE);
    CHECK_TRUE(written == sizeof(pointer_mode));

    CHECK_CUBLAS(cublasLtMatmulDescDestroy(desc));
}

static void test_matmul_preference_attributes() {
    cublasLtMatmulPreference_t pref = NULL;
    CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&pref));

    size_t workspace = 4096;
    CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
        pref,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspace,
        sizeof(workspace)));

    size_t round_trip = 0;
    size_t written = 0;
    CHECK_CUBLAS(cublasLtMatmulPreferenceGetAttribute(
        pref,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &round_trip,
        sizeof(round_trip),
        &written));
    CHECK_TRUE(round_trip == workspace);
    CHECK_TRUE(written == sizeof(round_trip));

    CHECK_CUBLAS(cublasLtMatmulPreferenceDestroy(pref));
}

static void test_matrix_layout_attributes() {
    cublasLtMatrixLayout_t layout = NULL;
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&layout, CUDA_R_32F, 8, 4, 8));

    uint64_t rows = 0;
    uint64_t cols = 0;
    int64_t ld = 0;
    size_t written = 0;

    CHECK_CUBLAS(cublasLtMatrixLayoutGetAttribute(
        layout,
        CUBLASLT_MATRIX_LAYOUT_ROWS,
        &rows,
        sizeof(rows),
        &written));
    CHECK_TRUE(rows == 8);
    CHECK_TRUE(written == sizeof(rows));

    CHECK_CUBLAS(cublasLtMatrixLayoutGetAttribute(
        layout,
        CUBLASLT_MATRIX_LAYOUT_COLS,
        &cols,
        sizeof(cols),
        &written));
    CHECK_TRUE(cols == 4);
    CHECK_TRUE(written == sizeof(cols));

    CHECK_CUBLAS(cublasLtMatrixLayoutGetAttribute(
        layout,
        CUBLASLT_MATRIX_LAYOUT_LD,
        &ld,
        sizeof(ld),
        &written));
    CHECK_TRUE(ld == 8);
    CHECK_TRUE(written == sizeof(ld));

    int32_t batch_count = 3;
    CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(
        layout,
        CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &batch_count,
        sizeof(batch_count)));

    batch_count = 0;
    CHECK_CUBLAS(cublasLtMatrixLayoutGetAttribute(
        layout,
        CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        &batch_count,
        sizeof(batch_count),
        &written));
    CHECK_TRUE(batch_count == 3);
    CHECK_TRUE(written == sizeof(batch_count));

    int64_t stride = 64;
    CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(
        layout,
        CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &stride,
        sizeof(stride)));

    stride = 0;
    CHECK_CUBLAS(cublasLtMatrixLayoutGetAttribute(
        layout,
        CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        &stride,
        sizeof(stride),
        &written));
    CHECK_TRUE(stride == 64);
    CHECK_TRUE(written == sizeof(stride));

    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(layout));
}

int main() {
    test_property_queries();
    test_version_and_status_helpers();
    test_heuristics_cache_capacity();
    test_matmul_desc_attributes();
    test_matmul_preference_attributes();
    test_matrix_layout_attributes();
    printf("cuBLASLt descriptor API test passed\n");
    return 0;
}
