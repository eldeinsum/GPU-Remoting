#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifdef cublasGetVersion
#undef cublasGetVersion
#endif
extern "C" cublasStatus_t cublasGetVersion(int *version);

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

static void expect_true(bool cond, const char *label) {
    if (!cond) {
        std::fprintf(stderr, "%s failed\n", label);
        std::exit(1);
    }
}

static void test_version_and_status(cublasHandle_t handle) {
    int legacy_version = 0;
    int handle_version = 0;
    CHECK_CUBLAS(cublasGetVersion(&legacy_version));
    CHECK_CUBLAS(cublasGetVersion_v2(handle, &handle_version));
    expect_true(legacy_version > 0, "legacy version");
    expect_true(handle_version > 0, "handle version");

    int major = 0;
    int minor = 0;
    int patch = 0;
    CHECK_CUBLAS(cublasGetProperty(MAJOR_VERSION, &major));
    CHECK_CUBLAS(cublasGetProperty(MINOR_VERSION, &minor));
    CHECK_CUBLAS(cublasGetProperty(PATCH_LEVEL, &patch));
    expect_true(major > 0, "major version property");
    expect_true(minor >= 0 && patch >= 0, "minor/patch version property");

    size_t cudart_version = cublasGetCudartVersion();
    expect_true(cudart_version > 0, "cudart version");

    const char *name = cublasGetStatusName(CUBLAS_STATUS_SUCCESS);
    const char *desc = cublasGetStatusString(CUBLAS_STATUS_SUCCESS);
    expect_true(name && std::strcmp(name, "CUBLAS_STATUS_SUCCESS") == 0,
                "status name");
    expect_true(desc && std::strlen(desc) > 0, "status string");
}

static void test_stream_and_modes(cublasHandle_t handle) {
    cudaStream_t initial_stream = reinterpret_cast<cudaStream_t>(0x1);
    CHECK_CUBLAS(cublasGetStream(handle, &initial_stream));
    expect_true(initial_stream == nullptr, "initial stream");

    cudaStream_t stream = nullptr;
    CHECK_CUDA(cudaStreamCreate(&stream));
    CHECK_CUBLAS(cublasSetStream(handle, stream));

    cudaStream_t queried_stream = nullptr;
    CHECK_CUBLAS(cublasGetStream(handle, &queried_stream));
    expect_true(queried_stream == stream, "queried stream");

    cublasAtomicsMode_t atomics = CUBLAS_ATOMICS_NOT_ALLOWED;
    CHECK_CUBLAS(cublasGetAtomicsMode(handle, &atomics));
    CHECK_CUBLAS(cublasSetAtomicsMode(handle, CUBLAS_ATOMICS_ALLOWED));
    cublasAtomicsMode_t atomics_after = CUBLAS_ATOMICS_NOT_ALLOWED;
    CHECK_CUBLAS(cublasGetAtomicsMode(handle, &atomics_after));
    expect_true(atomics_after == CUBLAS_ATOMICS_ALLOWED, "atomics mode");
    CHECK_CUBLAS(cublasSetAtomicsMode(handle, atomics));

    cublasMath_t math_mode = CUBLAS_DEFAULT_MATH;
    CHECK_CUBLAS(cublasGetMathMode(handle, &math_mode));
    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
    cublasMath_t math_after = CUBLAS_TENSOR_OP_MATH;
    CHECK_CUBLAS(cublasGetMathMode(handle, &math_after));
    expect_true(math_after == CUBLAS_DEFAULT_MATH, "math mode");
    CHECK_CUBLAS(cublasSetMathMode(handle, math_mode));

    int sm_target = 0;
    cublasStatus_t get_sm = cublasGetSmCountTarget(handle, &sm_target);
    if (get_sm == CUBLAS_STATUS_SUCCESS) {
        CHECK_CUBLAS(cublasSetSmCountTarget(handle, sm_target));
    } else if (get_sm != CUBLAS_STATUS_NOT_SUPPORTED) {
        std::fprintf(stderr, "unexpected cublasGetSmCountTarget status: %d\n",
                     static_cast<int>(get_sm));
        std::exit(1);
    }

    CHECK_CUBLAS(cublasSetStream(handle, nullptr));
    CHECK_CUDA(cudaStreamDestroy(stream));
}

int main() {
    cublasHandle_t handle = nullptr;
    CHECK_CUBLAS(cublasCreate(&handle));
    test_version_and_status(handle);
    test_stream_and_modes(handle);
    CHECK_CUBLAS(cublasDestroy(handle));
    std::puts("cuBLAS management API test passed");
    return 0;
}
