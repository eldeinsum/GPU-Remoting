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

static void check_true(bool condition, const char *expr, const char *file, int line) {
    if (!condition) {
        fprintf(stderr, "check failed: %s at %s:%d\n", expr, file, line);
        exit(1);
    }
}

#define CHECK_TRUE(expr) check_true((expr), #expr, __FILE__, __LINE__)

static void test_algo_id_and_init(cublasLtHandle_t handle) {
    int algo_ids[64] = {0};
    int returned = 0;

    CHECK_CUBLAS(cublasLtMatmulAlgoGetIds(
        handle,
        CUBLAS_COMPUTE_32F,
        CUDA_R_32F,
        CUDA_R_32F,
        CUDA_R_32F,
        CUDA_R_32F,
        CUDA_R_32F,
        64,
        algo_ids,
        &returned));
    CHECK_TRUE(returned > 0);

    bool initialized = false;
    for (int i = 0; i < returned; ++i) {
        cublasLtMatmulAlgo_t algo;
        cublasStatus_t status = cublasLtMatmulAlgoInit(
            handle,
            CUBLAS_COMPUTE_32F,
            CUDA_R_32F,
            CUDA_R_32F,
            CUDA_R_32F,
            CUDA_R_32F,
            CUDA_R_32F,
            algo_ids[i],
            &algo);
        if (status == CUBLAS_STATUS_SUCCESS) {
            int config_id = -1;
            size_t written = 0;
            CHECK_CUBLAS(cublasLtMatmulAlgoConfigGetAttribute(
                &algo,
                CUBLASLT_ALGO_CONFIG_ID,
                &config_id,
                sizeof(config_id),
                &written));
            CHECK_TRUE(config_id == algo_ids[i]);
            CHECK_TRUE(written == sizeof(config_id));
            initialized = true;
            break;
        }
    }

    CHECK_TRUE(initialized);
}

static void create_matmul_descriptors(
    cublasLtMatmulDesc_t *operation_desc,
    cublasLtMatmulPreference_t *preference,
    cublasLtMatrixLayout_t *adesc,
    cublasLtMatrixLayout_t *bdesc,
    cublasLtMatrixLayout_t *cdesc) {
    const int m = 16;
    const int n = 16;
    const int k = 16;

    CHECK_CUBLAS(cublasLtMatmulDescCreate(operation_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

    cublasOperation_t trans = CUBLAS_OP_N;
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        *operation_desc,
        CUBLASLT_MATMUL_DESC_TRANSA,
        &trans,
        sizeof(trans)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        *operation_desc,
        CUBLASLT_MATMUL_DESC_TRANSB,
        &trans,
        sizeof(trans)));

    CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(preference));
    size_t workspace_limit = 0;
    CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
        *preference,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspace_limit,
        sizeof(workspace_limit)));

    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(adesc, CUDA_R_32F, m, k, m));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(bdesc, CUDA_R_32F, k, n, k));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(cdesc, CUDA_R_32F, m, n, m));
}

static void destroy_matmul_descriptors(
    cublasLtMatmulDesc_t operation_desc,
    cublasLtMatmulPreference_t preference,
    cublasLtMatrixLayout_t adesc,
    cublasLtMatrixLayout_t bdesc,
    cublasLtMatrixLayout_t cdesc) {
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(adesc));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(bdesc));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(cdesc));
    CHECK_CUBLAS(cublasLtMatmulPreferenceDestroy(preference));
    CHECK_CUBLAS(cublasLtMatmulDescDestroy(operation_desc));
}

static void test_heuristic_algo_attributes(cublasLtHandle_t handle) {
    cublasLtMatmulDesc_t operation_desc = NULL;
    cublasLtMatmulPreference_t preference = NULL;
    cublasLtMatrixLayout_t adesc = NULL;
    cublasLtMatrixLayout_t bdesc = NULL;
    cublasLtMatrixLayout_t cdesc = NULL;

    create_matmul_descriptors(&operation_desc, &preference, &adesc, &bdesc, &cdesc);

    cublasLtMatmulHeuristicResult_t heuristic;
    int returned = 0;
    CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(
        handle,
        operation_desc,
        adesc,
        bdesc,
        cdesc,
        cdesc,
        preference,
        1,
        &heuristic,
        &returned));
    CHECK_TRUE(returned == 1);
    CHECK_TRUE(heuristic.state == CUBLAS_STATUS_SUCCESS);

    uint32_t pointer_mode_mask = 0;
    size_t written = 0;
    CHECK_CUBLAS(cublasLtMatmulAlgoCapGetAttribute(
        &heuristic.algo,
        CUBLASLT_ALGO_CAP_POINTER_MODE_MASK,
        &pointer_mode_mask,
        sizeof(pointer_mode_mask),
        &written));
    CHECK_TRUE(pointer_mode_mask != 0);
    CHECK_TRUE(written == sizeof(pointer_mode_mask));

    int config_id = -1;
    written = 0;
    CHECK_CUBLAS(cublasLtMatmulAlgoConfigGetAttribute(
        &heuristic.algo,
        CUBLASLT_ALGO_CONFIG_ID,
        &config_id,
        sizeof(config_id),
        &written));
    CHECK_TRUE(config_id >= 0);
    CHECK_TRUE(written == sizeof(config_id));

    destroy_matmul_descriptors(operation_desc, preference, adesc, bdesc, cdesc);
}

int main() {
    cublasLtHandle_t handle = NULL;
    CHECK_CUBLAS(cublasLtCreate(&handle));

    test_algo_id_and_init(handle);
    test_heuristic_algo_attributes(handle);

    CHECK_CUBLAS(cublasLtDestroy(handle));
    printf("cuBLASLt algorithm API test passed\n");
    return 0;
}
