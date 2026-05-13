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
    expect_close(got.x, real, 1e-10, label);
    expect_close(got.y, imag, 1e-10, label);
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
        expect_close(got[i], expected[i], 1e-10, label);
}

static void test_rotg() {
    float sa = 3.0f;
    float sb = 4.0f;
    float sc = 0.0f;
    float ss = 0.0f;
    cublasSrotg(&sa, &sb, &sc, &ss);
    expect_close(sa, 5.0, 1e-4, "cublasSrotg a");
    expect_close(sc, 0.6, 1e-4, "cublasSrotg c");
    expect_close(ss, 0.8, 1e-4, "cublasSrotg s");

    double da = 3.0;
    double db = 4.0;
    double dc = 0.0;
    double ds = 0.0;
    cublasDrotg(&da, &db, &dc, &ds);
    expect_close(da, 5.0, 1e-10, "cublasDrotg a");
    expect_close(dc, 0.6, 1e-10, "cublasDrotg c");
    expect_close(ds, 0.8, 1e-10, "cublasDrotg s");

    cuComplex ca = make_cuComplex(3.0f, 0.0f);
    cuComplex cb = make_cuComplex(4.0f, 0.0f);
    float csc = 0.0f;
    cuComplex ccs = make_cuComplex(0.0f, 0.0f);
    cublasCrotg(&ca, cb, &csc, &ccs);
    expect_complex(ca, 5.0f, 0.0f, "cublasCrotg a");
    expect_close(csc, 0.6, 1e-4, "cublasCrotg c");
    expect_complex(ccs, 0.8f, 0.0f, "cublasCrotg s");

    cuDoubleComplex za = make_cuDoubleComplex(3.0, 0.0);
    cuDoubleComplex zb = make_cuDoubleComplex(4.0, 0.0);
    double zsc = 0.0;
    cuDoubleComplex zcs = make_cuDoubleComplex(0.0, 0.0);
    cublasZrotg(&za, zb, &zsc, &zcs);
    expect_complex(za, 5.0, 0.0, "cublasZrotg a");
    expect_close(zsc, 0.6, 1e-10, "cublasZrotg c");
    expect_complex(zcs, 0.8, 0.0, "cublasZrotg s");
}

static void test_rotm() {
    float *sx = legacy_alloc<float>(2);
    float *sy = legacy_alloc<float>(2);
    set_vec(std::vector<float>{1.0f, 2.0f}, sx);
    set_vec(std::vector<float>{3.0f, 4.0f}, sy);
    const float sparam[5] = {-1.0f, 2.0f, 3.0f, 5.0f, 7.0f};
    CHECK_LAST(cublasSrotm(2, sx, 1, sy, 1, sparam));
    expect_vec(get_vec(sx, 2), {17.0f, 24.0f}, "cublasSrotm x");
    expect_vec(get_vec(sy, 2), {24.0f, 34.0f}, "cublasSrotm y");
    CHECK_CUBLAS(cublasFree(sx));
    CHECK_CUBLAS(cublasFree(sy));

    double *dx = legacy_alloc<double>(2);
    double *dy = legacy_alloc<double>(2);
    set_vec(std::vector<double>{1.0, 2.0}, dx);
    set_vec(std::vector<double>{3.0, 4.0}, dy);
    const double dparam[5] = {-1.0, 2.0, 3.0, 5.0, 7.0};
    CHECK_LAST(cublasDrotm(2, dx, 1, dy, 1, dparam));
    expect_vec(get_vec(dx, 2), {17.0, 24.0}, "cublasDrotm x");
    expect_vec(get_vec(dy, 2), {24.0, 34.0}, "cublasDrotm y");
    CHECK_CUBLAS(cublasFree(dx));
    CHECK_CUBLAS(cublasFree(dy));
}

static void test_rotmg() {
    float sd1 = 1.0f;
    float sd2 = 1.0f;
    float sx1 = 1.0f;
    const float sy1 = 0.0f;
    float sparam[5] = {99.0f, 98.0f, 97.0f, 96.0f, 95.0f};
    cublasSrotmg(&sd1, &sd2, &sx1, &sy1, sparam);
    expect_close(sd1, 1.0, 1e-4, "cublasSrotmg d1");
    expect_close(sd2, 1.0, 1e-4, "cublasSrotmg d2");
    expect_close(sx1, 1.0, 1e-4, "cublasSrotmg x1");
    expect_close(sparam[0], -2.0, 1e-4, "cublasSrotmg flag");

    double dd1 = 1.0;
    double dd2 = 1.0;
    double dx1 = 1.0;
    const double dy1 = 0.0;
    double dparam[5] = {99.0, 98.0, 97.0, 96.0, 95.0};
    cublasDrotmg(&dd1, &dd2, &dx1, &dy1, dparam);
    expect_close(dd1, 1.0, 1e-10, "cublasDrotmg d1");
    expect_close(dd2, 1.0, 1e-10, "cublasDrotmg d2");
    expect_close(dx1, 1.0, 1e-10, "cublasDrotmg x1");
    expect_close(dparam[0], -2.0, 1e-10, "cublasDrotmg flag");
}

int main() {
    test_rotg();
    test_rotm();
    test_rotmg();
    std::puts("legacy cuBLAS rotation parameter API test passed");
    return 0;
}
