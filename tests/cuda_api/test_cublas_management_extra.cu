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

static void test_emulation_controls(cublasHandle_t handle) {
    cublasEmulationStrategy_t strategy = CUBLAS_EMULATION_STRATEGY_DEFAULT;
    cublasStatus_t status = cublasGetEmulationStrategy(handle, &strategy);
    if (status == CUBLAS_STATUS_NOT_SUPPORTED) {
        return;
    }
    CHECK_CUBLAS(status);
    CHECK_CUBLAS(cublasSetEmulationStrategy(
        handle, CUBLAS_EMULATION_STRATEGY_PERFORMANT));
    cublasEmulationStrategy_t strategy_after = CUBLAS_EMULATION_STRATEGY_DEFAULT;
    CHECK_CUBLAS(cublasGetEmulationStrategy(handle, &strategy_after));
    expect_true(strategy_after == CUBLAS_EMULATION_STRATEGY_PERFORMANT,
                "emulation strategy");
    CHECK_CUBLAS(cublasSetEmulationStrategy(handle, strategy));

    cudaEmulationSpecialValuesSupport special_values =
        CUDA_EMULATION_SPECIAL_VALUES_SUPPORT_DEFAULT;
    CHECK_CUBLAS(cublasGetEmulationSpecialValuesSupport(handle,
                                                        &special_values));
    CHECK_CUBLAS(cublasSetEmulationSpecialValuesSupport(
        handle, CUDA_EMULATION_SPECIAL_VALUES_SUPPORT_NONE));
    cudaEmulationSpecialValuesSupport special_values_after =
        CUDA_EMULATION_SPECIAL_VALUES_SUPPORT_DEFAULT;
    CHECK_CUBLAS(cublasGetEmulationSpecialValuesSupport(
        handle, &special_values_after));
    expect_true(special_values_after == CUDA_EMULATION_SPECIAL_VALUES_SUPPORT_NONE,
                "emulation special values");
    CHECK_CUBLAS(cublasSetEmulationSpecialValuesSupport(handle, special_values));

    cudaEmulationMantissaControl mantissa_control =
        CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC;
    CHECK_CUBLAS(cublasGetFixedPointEmulationMantissaControl(
        handle, &mantissa_control));
    CHECK_CUBLAS(cublasSetFixedPointEmulationMantissaControl(
        handle, CUDA_EMULATION_MANTISSA_CONTROL_FIXED));
    cudaEmulationMantissaControl mantissa_control_after =
        CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC;
    CHECK_CUBLAS(cublasGetFixedPointEmulationMantissaControl(
        handle, &mantissa_control_after));
    expect_true(mantissa_control_after == CUDA_EMULATION_MANTISSA_CONTROL_FIXED,
                "mantissa control");
    CHECK_CUBLAS(cublasSetFixedPointEmulationMantissaControl(
        handle, mantissa_control));

    int max_mantissa_bits = 0;
    CHECK_CUBLAS(cublasGetFixedPointEmulationMaxMantissaBitCount(
        handle, &max_mantissa_bits));
    CHECK_CUBLAS(cublasSetFixedPointEmulationMaxMantissaBitCount(
        handle, max_mantissa_bits));
    int max_mantissa_bits_after = -1;
    CHECK_CUBLAS(cublasGetFixedPointEmulationMaxMantissaBitCount(
        handle, &max_mantissa_bits_after));
    expect_true(max_mantissa_bits_after == max_mantissa_bits,
                "max mantissa bit count");

    int mantissa_bit_offset = 0;
    CHECK_CUBLAS(cublasGetFixedPointEmulationMantissaBitOffset(
        handle, &mantissa_bit_offset));
    CHECK_CUBLAS(cublasSetFixedPointEmulationMantissaBitOffset(
        handle, mantissa_bit_offset));
    int mantissa_bit_offset_after = -1;
    CHECK_CUBLAS(cublasGetFixedPointEmulationMantissaBitOffset(
        handle, &mantissa_bit_offset_after));
    expect_true(mantissa_bit_offset_after == mantissa_bit_offset,
                "mantissa bit offset");

    int *initial_bit_count_ptr = nullptr;
    CHECK_CUBLAS(cublasGetFixedPointEmulationMantissaBitCountPointer(
        handle, &initial_bit_count_ptr));
    int *device_bit_count = nullptr;
    CHECK_CUDA(cudaMalloc(&device_bit_count, sizeof(int)));
    CHECK_CUBLAS(cublasSetFixedPointEmulationMantissaBitCountPointer(
        handle, device_bit_count));
    int *queried_bit_count_ptr = nullptr;
    CHECK_CUBLAS(cublasGetFixedPointEmulationMantissaBitCountPointer(
        handle, &queried_bit_count_ptr));
    expect_true(queried_bit_count_ptr == device_bit_count,
                "mantissa bit count pointer");
    CHECK_CUBLAS(cublasSetFixedPointEmulationMantissaBitCountPointer(
        handle, initial_bit_count_ptr));
    CHECK_CUDA(cudaFree(device_bit_count));
}

int main() {
    cublasHandle_t handle = nullptr;
    CHECK_CUBLAS(cublasCreate(&handle));
    test_version_and_status(handle);
    test_stream_and_modes(handle);
    test_emulation_controls(handle);
    CHECK_CUBLAS(cublasDestroy(handle));
    std::puts("cuBLAS management API test passed");
    return 0;
}
