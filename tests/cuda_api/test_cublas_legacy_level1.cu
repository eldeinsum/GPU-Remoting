#include <cublas.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK_CUDA(expr)                                                       \
    do {                                                                       \
        cudaError_t status = (expr);                                           \
        if (status != cudaSuccess) {                                           \
            std::fprintf(stderr, "%s failed: %s\n", #expr,                    \
                         cudaGetErrorString(status));                          \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

#define CHECK_CUBLAS(expr)                                                     \
    do {                                                                       \
        cublasStatus_t status = (expr);                                        \
        if (status != CUBLAS_STATUS_SUCCESS) {                                 \
            std::fprintf(stderr, "%s failed: %d\n", #expr,                    \
                         static_cast<int>(status));                            \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

#define CHECK_LAST(expr)                                                       \
    do {                                                                       \
        (expr);                                                                \
        CHECK_CUBLAS(cublasGetError());                                        \
    } while (0)

static void expect_close(double got, double expected, double eps,
                         const char *label) {
    if (std::fabs(got - expected) > eps) {
        std::fprintf(stderr, "%s got %.12f expected %.12f\n", label, got,
                     expected);
        std::exit(1);
    }
}

static void expect_complex(cuComplex got, float real, float imag,
                           const char *label) {
    expect_close(got.x, real, 1e-4, label);
    expect_close(got.y, imag, 1e-4, label);
}

static void expect_complex(cuDoubleComplex got, double real, double imag,
                           const char *label) {
    expect_close(got.x, real, 1e-9, label);
    expect_close(got.y, imag, 1e-9, label);
}

template <typename T>
static T *legacy_alloc(int n) {
    void *ptr = nullptr;
    CHECK_CUBLAS(cublasAlloc(n, sizeof(T), &ptr));
    return static_cast<T *>(ptr);
}

template <typename T>
static void set_vec(const std::vector<T> &host, T *device) {
    CHECK_CUBLAS(cublasSetVector(static_cast<int>(host.size()), sizeof(T),
                                 host.data(), 1, device, 1));
}

template <typename T>
static std::vector<T> get_vec(T *device, int n) {
    std::vector<T> out(static_cast<size_t>(n));
    CHECK_CUBLAS(cublasGetVector(n, sizeof(T), device, 1, out.data(), 1));
    return out;
}

static void expect_vec(const std::vector<float> &got,
                       const std::vector<float> &expected, const char *label) {
    for (size_t i = 0; i < expected.size(); ++i)
        expect_close(got[i], expected[i], 1e-4, label);
}

static void expect_vec(const std::vector<double> &got,
                       const std::vector<double> &expected, const char *label) {
    for (size_t i = 0; i < expected.size(); ++i)
        expect_close(got[i], expected[i], 1e-9, label);
}

static void expect_vec(const std::vector<cuComplex> &got,
                       const std::vector<cuComplex> &expected,
                       const char *label) {
    for (size_t i = 0; i < expected.size(); ++i)
        expect_complex(got[i], expected[i].x, expected[i].y, label);
}

static void expect_vec(const std::vector<cuDoubleComplex> &got,
                       const std::vector<cuDoubleComplex> &expected,
                       const char *label) {
    for (size_t i = 0; i < expected.size(); ++i)
        expect_complex(got[i], expected[i].x, expected[i].y, label);
}

static void test_management() {
    int version = 0;
    CHECK_CUBLAS(cublasGetVersion(&version));
    if (version <= 0) {
        std::fprintf(stderr, "invalid legacy cuBLAS version\n");
        std::exit(1);
    }

    CHECK_CUBLAS(cublasLoggerConfigure(0, 0, 0, nullptr));

    cudaStream_t stream = nullptr;
    CHECK_CUDA(cudaStreamCreate(&stream));
    CHECK_CUBLAS(cublasSetKernelStream(stream));
    CHECK_CUBLAS(cublasGetError());
    CHECK_CUBLAS(cublasSetKernelStream(nullptr));
    CHECK_CUBLAS(cublasGetError());
    CHECK_CUDA(cudaStreamDestroy(stream));
}

static void test_float_level1() {
    const int n = 3;
    float *x = legacy_alloc<float>(n);
    float *y = legacy_alloc<float>(n);
    float *z = legacy_alloc<float>(n);

    set_vec(std::vector<float>{1.0f, -2.0f, 3.0f}, x);
    set_vec(std::vector<float>{4.0f, 5.0f, -6.0f}, y);

    expect_close(cublasSnrm2(n, x, 1), std::sqrt(14.0), 1e-4,
                 "cublasSnrm2");
    CHECK_CUBLAS(cublasGetError());
    expect_close(cublasSdot(n, x, 1, y, 1), -24.0, 1e-4, "cublasSdot");
    CHECK_CUBLAS(cublasGetError());
    expect_close(cublasSasum(n, x, 1), 6.0, 1e-4, "cublasSasum");
    CHECK_CUBLAS(cublasGetError());
    if (cublasIsamax(n, x, 1) != 3 || cublasIsamin(n, x, 1) != 1) {
        std::fprintf(stderr, "float legacy index reduction mismatch\n");
        std::exit(1);
    }
    CHECK_CUBLAS(cublasGetError());

    CHECK_LAST(cublasScopy(n, x, 1, z, 1));
    expect_vec(get_vec(z, n), {1.0f, -2.0f, 3.0f}, "cublasScopy");
    CHECK_LAST(cublasSscal(n, 2.0f, z, 1));
    expect_vec(get_vec(z, n), {2.0f, -4.0f, 6.0f}, "cublasSscal");
    CHECK_LAST(cublasSaxpy(n, 0.5f, x, 1, y, 1));
    expect_vec(get_vec(y, n), {4.5f, 4.0f, -4.5f}, "cublasSaxpy");
    CHECK_LAST(cublasSswap(n, x, 1, y, 1));
    expect_vec(get_vec(x, n), {4.5f, 4.0f, -4.5f}, "cublasSswap x");
    expect_vec(get_vec(y, n), {1.0f, -2.0f, 3.0f}, "cublasSswap y");

    set_vec(std::vector<float>{1.0f, 0.0f}, x);
    set_vec(std::vector<float>{0.0f, 1.0f}, y);
    CHECK_LAST(cublasSrot(2, x, 1, y, 1, 0.0f, 1.0f));
    expect_vec(get_vec(x, 2), {0.0f, 1.0f}, "cublasSrot x");
    expect_vec(get_vec(y, 2), {-1.0f, 0.0f}, "cublasSrot y");

    CHECK_CUBLAS(cublasFree(x));
    CHECK_CUBLAS(cublasFree(y));
    CHECK_CUBLAS(cublasFree(z));
}

static void test_double_level1() {
    const int n = 3;
    double *x = legacy_alloc<double>(n);
    double *y = legacy_alloc<double>(n);
    double *z = legacy_alloc<double>(n);

    set_vec(std::vector<double>{1.0, -2.0, 3.0}, x);
    set_vec(std::vector<double>{4.0, 5.0, -6.0}, y);

    expect_close(cublasDnrm2(n, x, 1), std::sqrt(14.0), 1e-9,
                 "cublasDnrm2");
    CHECK_CUBLAS(cublasGetError());
    expect_close(cublasDdot(n, x, 1, y, 1), -24.0, 1e-9, "cublasDdot");
    CHECK_CUBLAS(cublasGetError());
    expect_close(cublasDasum(n, x, 1), 6.0, 1e-9, "cublasDasum");
    CHECK_CUBLAS(cublasGetError());
    if (cublasIdamax(n, x, 1) != 3 || cublasIdamin(n, x, 1) != 1) {
        std::fprintf(stderr, "double legacy index reduction mismatch\n");
        std::exit(1);
    }
    CHECK_CUBLAS(cublasGetError());

    CHECK_LAST(cublasDcopy(n, x, 1, z, 1));
    expect_vec(get_vec(z, n), {1.0, -2.0, 3.0}, "cublasDcopy");
    CHECK_LAST(cublasDscal(n, 2.0, z, 1));
    expect_vec(get_vec(z, n), {2.0, -4.0, 6.0}, "cublasDscal");
    CHECK_LAST(cublasDaxpy(n, 0.5, x, 1, y, 1));
    expect_vec(get_vec(y, n), {4.5, 4.0, -4.5}, "cublasDaxpy");
    CHECK_LAST(cublasDswap(n, x, 1, y, 1));
    expect_vec(get_vec(x, n), {4.5, 4.0, -4.5}, "cublasDswap x");
    expect_vec(get_vec(y, n), {1.0, -2.0, 3.0}, "cublasDswap y");

    set_vec(std::vector<double>{1.0, 0.0}, x);
    set_vec(std::vector<double>{0.0, 1.0}, y);
    CHECK_LAST(cublasDrot(2, x, 1, y, 1, 0.0, 1.0));
    expect_vec(get_vec(x, 2), {0.0, 1.0}, "cublasDrot x");
    expect_vec(get_vec(y, 2), {-1.0, 0.0}, "cublasDrot y");

    CHECK_CUBLAS(cublasFree(x));
    CHECK_CUBLAS(cublasFree(y));
    CHECK_CUBLAS(cublasFree(z));
}

static void test_complex_float_level1() {
    const int n = 2;
    cuComplex *x = legacy_alloc<cuComplex>(n);
    cuComplex *y = legacy_alloc<cuComplex>(n);
    cuComplex *z = legacy_alloc<cuComplex>(n);

    set_vec(std::vector<cuComplex>{make_cuComplex(1.0f, 2.0f),
                                   make_cuComplex(-3.0f, 4.0f)},
            x);
    set_vec(std::vector<cuComplex>{make_cuComplex(5.0f, -1.0f),
                                   make_cuComplex(2.0f, 3.0f)},
            y);

    expect_close(cublasScnrm2(n, x, 1), std::sqrt(30.0), 1e-4,
                 "cublasScnrm2");
    CHECK_CUBLAS(cublasGetError());
    expect_close(cublasScasum(n, x, 1), 10.0, 1e-4, "cublasScasum");
    CHECK_CUBLAS(cublasGetError());
    expect_complex(cublasCdotu(n, x, 1, y, 1), -11.0f, 8.0f, "cublasCdotu");
    CHECK_CUBLAS(cublasGetError());
    expect_complex(cublasCdotc(n, x, 1, y, 1), 9.0f, -28.0f,
                   "cublasCdotc");
    CHECK_CUBLAS(cublasGetError());
    if (cublasIcamax(n, x, 1) != 2 || cublasIcamin(n, x, 1) != 1) {
        std::fprintf(stderr, "complex float index reduction mismatch\n");
        std::exit(1);
    }
    CHECK_CUBLAS(cublasGetError());

    CHECK_LAST(cublasCcopy(n, x, 1, z, 1));
    expect_vec(get_vec(z, n), {make_cuComplex(1.0f, 2.0f),
                               make_cuComplex(-3.0f, 4.0f)},
               "cublasCcopy");
    CHECK_LAST(cublasCsscal(n, 2.0f, z, 1));
    expect_vec(get_vec(z, n), {make_cuComplex(2.0f, 4.0f),
                               make_cuComplex(-6.0f, 8.0f)},
               "cublasCsscal");
    CHECK_LAST(cublasCscal(n, make_cuComplex(0.0f, 1.0f), z, 1));
    expect_vec(get_vec(z, n), {make_cuComplex(-4.0f, 2.0f),
                               make_cuComplex(-8.0f, -6.0f)},
               "cublasCscal");
    CHECK_LAST(cublasCaxpy(n, make_cuComplex(1.0f, 0.0f), x, 1, y, 1));
    expect_vec(get_vec(y, n), {make_cuComplex(6.0f, 1.0f),
                               make_cuComplex(-1.0f, 7.0f)},
               "cublasCaxpy");
    CHECK_LAST(cublasCswap(n, x, 1, y, 1));
    expect_vec(get_vec(x, n), {make_cuComplex(6.0f, 1.0f),
                               make_cuComplex(-1.0f, 7.0f)},
               "cublasCswap x");

    CHECK_LAST(cublasCrot(n, x, 1, y, 1, 1.0f, make_cuComplex(0.0f, 0.0f)));
    CHECK_LAST(cublasCsrot(n, x, 1, y, 1, 1.0f, 0.0f));

    CHECK_CUBLAS(cublasFree(x));
    CHECK_CUBLAS(cublasFree(y));
    CHECK_CUBLAS(cublasFree(z));
}

static void test_complex_double_level1() {
    const int n = 1;
    cuDoubleComplex *x = legacy_alloc<cuDoubleComplex>(n);
    cuDoubleComplex *y = legacy_alloc<cuDoubleComplex>(n);
    cuDoubleComplex *z = legacy_alloc<cuDoubleComplex>(n);

    set_vec(std::vector<cuDoubleComplex>{make_cuDoubleComplex(1.0, 2.0)}, x);
    set_vec(std::vector<cuDoubleComplex>{make_cuDoubleComplex(3.0, 4.0)}, y);

    expect_close(cublasDznrm2(n, x, 1), std::sqrt(5.0), 1e-9,
                 "cublasDznrm2");
    CHECK_CUBLAS(cublasGetError());
    expect_close(cublasDzasum(n, x, 1), 3.0, 1e-9, "cublasDzasum");
    CHECK_CUBLAS(cublasGetError());
    expect_complex(cublasZdotu(n, x, 1, y, 1), -5.0, 10.0, "cublasZdotu");
    CHECK_CUBLAS(cublasGetError());
    expect_complex(cublasZdotc(n, x, 1, y, 1), 11.0, -2.0, "cublasZdotc");
    CHECK_CUBLAS(cublasGetError());
    if (cublasIzamax(n, x, 1) != 1 || cublasIzamin(n, x, 1) != 1) {
        std::fprintf(stderr, "complex double index reduction mismatch\n");
        std::exit(1);
    }
    CHECK_CUBLAS(cublasGetError());

    CHECK_LAST(cublasZcopy(n, x, 1, z, 1));
    expect_vec(get_vec(z, n), {make_cuDoubleComplex(1.0, 2.0)},
               "cublasZcopy");
    CHECK_LAST(cublasZdscal(n, 2.0, z, 1));
    expect_vec(get_vec(z, n), {make_cuDoubleComplex(2.0, 4.0)},
               "cublasZdscal");
    CHECK_LAST(cublasZscal(n, make_cuDoubleComplex(0.0, 1.0), z, 1));
    expect_vec(get_vec(z, n), {make_cuDoubleComplex(-4.0, 2.0)},
               "cublasZscal");
    CHECK_LAST(cublasZaxpy(n, make_cuDoubleComplex(1.0, 0.0), x, 1, y, 1));
    expect_vec(get_vec(y, n), {make_cuDoubleComplex(4.0, 6.0)},
               "cublasZaxpy");
    CHECK_LAST(cublasZswap(n, x, 1, y, 1));
    expect_vec(get_vec(x, n), {make_cuDoubleComplex(4.0, 6.0)},
               "cublasZswap x");

    CHECK_LAST(cublasZrot(n, x, 1, y, 1, 1.0,
                          make_cuDoubleComplex(0.0, 0.0)));
    CHECK_LAST(cublasZdrot(n, x, 1, y, 1, 1.0, 0.0));

    CHECK_CUBLAS(cublasFree(x));
    CHECK_CUBLAS(cublasFree(y));
    CHECK_CUBLAS(cublasFree(z));
}

int main() {
    CHECK_CUBLAS(cublasInit());
    test_management();
    test_float_level1();
    test_double_level1();
    test_complex_float_level1();
    test_complex_double_level1();
    CHECK_CUBLAS(cublasShutdown());
    std::puts("legacy cuBLAS Level-1 API test passed");
    return 0;
}
