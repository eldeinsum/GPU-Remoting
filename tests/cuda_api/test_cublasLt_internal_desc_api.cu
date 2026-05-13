#include <cublasLt.h>
#include <cuda_runtime.h>

#include <math.h>
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

static void logger_callback(int, const char *, const char *) {}

template <typename T, typename Desc, typename Attr>
static T get_attr(Desc desc, Attr attr) {
    T value{};
    size_t written = 0;
    CHECK_CUBLAS(cublasLtMatrixLayoutGetAttribute(desc, attr, &value, sizeof(value), &written));
    CHECK_TRUE(written == sizeof(value));
    return value;
}

static void check_close(float actual, float expected) {
    CHECK_TRUE(fabsf(actual - expected) < 1.0e-4f);
}

static void test_process_local_helpers() {
    CHECK_CUBLAS(cublasLtLoggerSetCallback(logger_callback));
    CHECK_CUBLAS(cublasLtLoggerSetCallback(NULL));

    FILE *file = tmpfile();
    CHECK_TRUE(file != NULL);
    CHECK_CUBLAS(cublasLtLoggerSetFile(file));
    fclose(file);
}

static void test_stack_descriptor_attributes() {
    cublasLtMatrixLayoutOpaque_t layout_storage{};
    cublasLtMatrixLayout_t layout = &layout_storage;
    CHECK_CUBLAS(cublasLtMatrixLayoutInit(layout, CUDA_R_32F, 8, 4, 8));

    size_t written = 0;
    uint64_t rows = 0;
    CHECK_CUBLAS(cublasLtMatrixLayoutGetAttribute(
        layout, CUBLASLT_MATRIX_LAYOUT_ROWS, &rows, sizeof(rows), &written));
    CHECK_TRUE(rows == 8);
    CHECK_TRUE(written == sizeof(rows));

    int32_t batch_count = 3;
    CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(
        layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));

    batch_count = 0;
    written = 0;
    CHECK_CUBLAS(cublasLtMatrixLayoutGetAttribute(
        layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count), &written));
    CHECK_TRUE(batch_count == 3);
    CHECK_TRUE(written == sizeof(batch_count));

    cublasLtMatmulDescOpaque_t op_storage{};
    cublasLtMatmulDesc_t op_desc = &op_storage;
    CHECK_CUBLAS(cublasLtMatmulDescInit(op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

    cublasLtPointerMode_t pointer_mode = CUBLASLT_POINTER_MODE_DEVICE;
    written = 0;
    CHECK_CUBLAS(cublasLtMatmulDescGetAttribute(
        op_desc, CUBLASLT_MATMUL_DESC_POINTER_MODE, &pointer_mode, sizeof(pointer_mode), &written));
    CHECK_TRUE(pointer_mode == CUBLASLT_POINTER_MODE_HOST);
    CHECK_TRUE(written == sizeof(pointer_mode));

    pointer_mode = CUBLASLT_POINTER_MODE_DEVICE;
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        op_desc, CUBLASLT_MATMUL_DESC_POINTER_MODE, &pointer_mode, sizeof(pointer_mode)));
    pointer_mode = CUBLASLT_POINTER_MODE_HOST;
    CHECK_CUBLAS(cublasLtMatmulDescGetAttribute(
        op_desc, CUBLASLT_MATMUL_DESC_POINTER_MODE, &pointer_mode, sizeof(pointer_mode), &written));
    CHECK_TRUE(pointer_mode == CUBLASLT_POINTER_MODE_DEVICE);

    cublasLtMatrixTransformDescOpaque_t transform_storage{};
    cublasLtMatrixTransformDesc_t transform_desc = &transform_storage;
    CHECK_CUBLAS(cublasLtMatrixTransformDescInit(transform_desc, CUDA_R_32F));
    pointer_mode = CUBLASLT_POINTER_MODE_DEVICE;
    CHECK_CUBLAS(cublasLtMatrixTransformDescSetAttribute(
        transform_desc,
        CUBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE,
        &pointer_mode,
        sizeof(pointer_mode)));
    pointer_mode = CUBLASLT_POINTER_MODE_HOST;
    CHECK_CUBLAS(cublasLtMatrixTransformDescGetAttribute(
        transform_desc,
        CUBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE,
        &pointer_mode,
        sizeof(pointer_mode),
        &written));
    CHECK_TRUE(pointer_mode == CUBLASLT_POINTER_MODE_DEVICE);

    cublasLtMatmulPreferenceOpaque_t pref_storage{};
    cublasLtMatmulPreference_t pref = &pref_storage;
    CHECK_CUBLAS(cublasLtMatmulPreferenceInit(pref));
    uint64_t workspace = 0;
    CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
        pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace, sizeof(workspace)));
    workspace = 1;
    CHECK_CUBLAS(cublasLtMatmulPreferenceGetAttribute(
        pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace, sizeof(workspace), &written));
    CHECK_TRUE(workspace == 0);

    cublasLtEmulationDescOpaque_t emulation_storage{};
    cublasLtEmulationDesc_t emulation_desc = &emulation_storage;
    CHECK_CUBLAS(cublasLtEmulationDescInit(emulation_desc));
    int value = 7;
    CHECK_CUBLAS(cublasLtEmulationDescSetAttribute(
        emulation_desc,
        CUBLASLT_EMULATION_DESC_FIXEDPOINT_MAX_MANTISSA_BIT_COUNT,
        &value,
        sizeof(value)));
    value = 0;
    CHECK_CUBLAS(cublasLtEmulationDescGetAttribute(
        emulation_desc,
        CUBLASLT_EMULATION_DESC_FIXEDPOINT_MAX_MANTISSA_BIT_COUNT,
        &value,
        sizeof(value),
        &written));
    CHECK_TRUE(value == 7);
}

static void test_stack_grouped_layout() {
    const int group_count = 2;
    const int64_t rows[] = {8, 16};
    const int64_t cols[] = {4, 8};
    const int64_t ld[] = {8, 16};

    int64_t *d_rows = NULL;
    int64_t *d_cols = NULL;
    int64_t *d_ld = NULL;
    CHECK_CUDA(cudaMalloc(&d_rows, sizeof(rows)));
    CHECK_CUDA(cudaMalloc(&d_cols, sizeof(cols)));
    CHECK_CUDA(cudaMalloc(&d_ld, sizeof(ld)));
    CHECK_CUDA(cudaMemcpy(d_rows, rows, sizeof(rows), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_cols, cols, sizeof(cols), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_ld, ld, sizeof(ld), cudaMemcpyHostToDevice));

    cublasLtMatrixLayoutOpaque_t grouped_storage{};
    cublasLtMatrixLayout_t grouped = &grouped_storage;
    CHECK_CUBLAS(cublasLtGroupedMatrixLayoutInit(
        grouped, CUDA_R_32F, group_count, d_rows, d_cols, d_ld));

    CHECK_TRUE(get_attr<cublasLtBatchMode_t>(
                   grouped, CUBLASLT_MATRIX_LAYOUT_BATCH_MODE) == CUBLASLT_BATCH_MODE_GROUPED);
    CHECK_TRUE(get_attr<int32_t>(
                   grouped, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT) == group_count);

    CHECK_CUDA(cudaFree(d_ld));
    CHECK_CUDA(cudaFree(d_cols));
    CHECK_CUDA(cudaFree(d_rows));
}

static void test_stack_matmul_and_transform() {
    const int m = 2;
    const int n = 2;
    const int k = 2;
    const float host_a[] = {1.0f, 0.0f, 0.0f, 1.0f};
    const float host_b[] = {5.0f, 6.0f, 7.0f, 8.0f};
    float host_c[4] = {};

    float *dev_a = NULL;
    float *dev_b = NULL;
    float *dev_c = NULL;
    CHECK_CUDA(cudaMalloc(&dev_a, sizeof(host_a)));
    CHECK_CUDA(cudaMalloc(&dev_b, sizeof(host_b)));
    CHECK_CUDA(cudaMalloc(&dev_c, sizeof(host_c)));
    CHECK_CUDA(cudaMemcpy(dev_a, host_a, sizeof(host_a), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dev_b, host_b, sizeof(host_b), cudaMemcpyHostToDevice));

    cublasLtHandle_t handle = NULL;
    CHECK_CUBLAS(cublasLtCreate(&handle));

    cublasLtMatmulDescOpaque_t op_storage{};
    cublasLtMatmulDesc_t op_desc = &op_storage;
    CHECK_CUBLAS(cublasLtMatmulDescInit(op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

    cublasLtMatrixLayoutOpaque_t a_storage{};
    cublasLtMatrixLayoutOpaque_t b_storage{};
    cublasLtMatrixLayoutOpaque_t c_storage{};
    cublasLtMatrixLayout_t a_desc = &a_storage;
    cublasLtMatrixLayout_t b_desc = &b_storage;
    cublasLtMatrixLayout_t c_desc = &c_storage;
    CHECK_CUBLAS(cublasLtMatrixLayoutInit(a_desc, CUDA_R_32F, m, k, m));
    CHECK_CUBLAS(cublasLtMatrixLayoutInit(b_desc, CUDA_R_32F, k, n, k));
    CHECK_CUBLAS(cublasLtMatrixLayoutInit(c_desc, CUDA_R_32F, m, n, m));

    cublasLtMatmulPreferenceOpaque_t pref_storage{};
    cublasLtMatmulPreference_t pref = &pref_storage;
    CHECK_CUBLAS(cublasLtMatmulPreferenceInit(pref));
    uint64_t workspace = 0;
    CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
        pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace, sizeof(workspace)));

    cublasLtMatmulHeuristicResult_t heuristic{};
    int returned = 0;
    CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(
        handle, op_desc, a_desc, b_desc, c_desc, c_desc, pref, 1, &heuristic, &returned));
    CHECK_TRUE(returned == 1);

    float alpha = 1.0f;
    float beta = 0.0f;
    CHECK_CUDA(cudaMemset(dev_c, 0, sizeof(host_c)));
    CHECK_CUBLAS(cublasLtMatmul(
        handle,
        op_desc,
        &alpha,
        dev_a,
        a_desc,
        dev_b,
        b_desc,
        &beta,
        dev_c,
        c_desc,
        dev_c,
        c_desc,
        &heuristic.algo,
        NULL,
        0,
        0));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(host_c, dev_c, sizeof(host_c), cudaMemcpyDeviceToHost));
    for (int i = 0; i < 4; ++i) {
        check_close(host_c[i], host_b[i]);
    }

    cublasLtMatrixTransformDescOpaque_t transform_storage{};
    cublasLtMatrixTransformDesc_t transform_desc = &transform_storage;
    CHECK_CUBLAS(cublasLtMatrixTransformDescInit(transform_desc, CUDA_R_32F));
    alpha = 2.0f;
    beta = 3.0f;
    CHECK_CUDA(cudaMemset(dev_c, 0, sizeof(host_c)));
    CHECK_CUBLAS(cublasLtMatrixTransform(
        handle, transform_desc, &alpha, dev_a, a_desc, &beta, dev_b, c_desc, dev_c, c_desc, 0));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(host_c, dev_c, sizeof(host_c), cudaMemcpyDeviceToHost));
    for (int i = 0; i < 4; ++i) {
        check_close(host_c[i], alpha * host_a[i] + beta * host_b[i]);
    }

    CHECK_CUBLAS(cublasLtDestroy(handle));
    CHECK_CUDA(cudaFree(dev_c));
    CHECK_CUDA(cudaFree(dev_b));
    CHECK_CUDA(cudaFree(dev_a));
}

int main() {
    test_process_local_helpers();
    test_stack_descriptor_attributes();
    test_stack_grouped_layout();
    test_stack_matmul_and_transform();
    printf("cuBLASLt internal descriptor API test passed\n");
    return 0;
}
