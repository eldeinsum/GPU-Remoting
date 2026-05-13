#include <cublas.h>
#include <cuComplex.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

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

template <typename T> struct Ops;

template <> struct Ops<float> {
    static float value(float real, float) { return real; }
    static float zero() { return 0.0f; }
    static float add(float a, float b) { return a + b; }
    static float mul(float a, float b) { return a * b; }
    static float conj(float a) { return a; }
    static float alpha() { return 0.75f; }
    static float beta() { return -0.5f; }
    static bool near(float a, float b) { return std::fabs(a - b) < 1.0e-4f; }
};

template <> struct Ops<double> {
    static double value(float real, float) { return static_cast<double>(real); }
    static double zero() { return 0.0; }
    static double add(double a, double b) { return a + b; }
    static double mul(double a, double b) { return a * b; }
    static double conj(double a) { return a; }
    static double alpha() { return 0.75; }
    static double beta() { return -0.5; }
    static bool near(double a, double b) { return std::fabs(a - b) < 1.0e-9; }
};

template <> struct Ops<cuComplex> {
    static cuComplex value(float real, float imag) {
        return make_cuComplex(real, imag);
    }
    static cuComplex zero() { return make_cuComplex(0.0f, 0.0f); }
    static cuComplex add(cuComplex a, cuComplex b) { return cuCaddf(a, b); }
    static cuComplex mul(cuComplex a, cuComplex b) { return cuCmulf(a, b); }
    static cuComplex conj(cuComplex a) { return cuConjf(a); }
    static cuComplex alpha() { return make_cuComplex(0.75f, -0.25f); }
    static cuComplex beta() { return make_cuComplex(-0.5f, 0.125f); }
    static bool near(cuComplex a, cuComplex b) {
        return std::fabs(a.x - b.x) < 1.0e-3f &&
               std::fabs(a.y - b.y) < 1.0e-3f;
    }
};

template <> struct Ops<cuDoubleComplex> {
    static cuDoubleComplex value(float real, float imag) {
        return make_cuDoubleComplex(static_cast<double>(real),
                                    static_cast<double>(imag));
    }
    static cuDoubleComplex zero() { return make_cuDoubleComplex(0.0, 0.0); }
    static cuDoubleComplex add(cuDoubleComplex a, cuDoubleComplex b) {
        return cuCadd(a, b);
    }
    static cuDoubleComplex mul(cuDoubleComplex a, cuDoubleComplex b) {
        return cuCmul(a, b);
    }
    static cuDoubleComplex conj(cuDoubleComplex a) { return cuConj(a); }
    static cuDoubleComplex alpha() { return make_cuDoubleComplex(0.75, -0.25); }
    static cuDoubleComplex beta() { return make_cuDoubleComplex(-0.5, 0.125); }
    static bool near(cuDoubleComplex a, cuDoubleComplex b) {
        return std::fabs(a.x - b.x) < 1.0e-9 &&
               std::fabs(a.y - b.y) < 1.0e-9;
    }
};

template <typename T> struct LegacyGemv;

template <> struct LegacyGemv<float> {
    static void call(int m, int n, float alpha, const float *A, int lda,
                     const float *x, float beta, float *y) {
        cublasSgemv('n', m, n, alpha, A, lda, x, 1, beta, y, 1);
    }
};

template <> struct LegacyGemv<double> {
    static void call(int m, int n, double alpha, const double *A, int lda,
                     const double *x, double beta, double *y) {
        cublasDgemv('n', m, n, alpha, A, lda, x, 1, beta, y, 1);
    }
};

template <> struct LegacyGemv<cuComplex> {
    static void call(int m, int n, cuComplex alpha, const cuComplex *A, int lda,
                     const cuComplex *x, cuComplex beta, cuComplex *y) {
        cublasCgemv('n', m, n, alpha, A, lda, x, 1, beta, y, 1);
    }
};

template <> struct LegacyGemv<cuDoubleComplex> {
    static void call(int m, int n, cuDoubleComplex alpha,
                     const cuDoubleComplex *A, int lda,
                     const cuDoubleComplex *x, cuDoubleComplex beta,
                     cuDoubleComplex *y) {
        cublasZgemv('n', m, n, alpha, A, lda, x, 1, beta, y, 1);
    }
};

template <typename T, bool Conjugate> struct LegacyGer;

template <> struct LegacyGer<float, false> {
    static void call(int m, int n, float alpha, const float *x, const float *y,
                     float *A, int lda) {
        cublasSger(m, n, alpha, x, 1, y, 1, A, lda);
    }
};

template <> struct LegacyGer<double, false> {
    static void call(int m, int n, double alpha, const double *x,
                     const double *y, double *A, int lda) {
        cublasDger(m, n, alpha, x, 1, y, 1, A, lda);
    }
};

template <> struct LegacyGer<cuComplex, false> {
    static void call(int m, int n, cuComplex alpha, const cuComplex *x,
                     const cuComplex *y, cuComplex *A, int lda) {
        cublasCgeru(m, n, alpha, x, 1, y, 1, A, lda);
    }
};

template <> struct LegacyGer<cuComplex, true> {
    static void call(int m, int n, cuComplex alpha, const cuComplex *x,
                     const cuComplex *y, cuComplex *A, int lda) {
        cublasCgerc(m, n, alpha, x, 1, y, 1, A, lda);
    }
};

template <> struct LegacyGer<cuDoubleComplex, false> {
    static void call(int m, int n, cuDoubleComplex alpha,
                     const cuDoubleComplex *x, const cuDoubleComplex *y,
                     cuDoubleComplex *A, int lda) {
        cublasZgeru(m, n, alpha, x, 1, y, 1, A, lda);
    }
};

template <> struct LegacyGer<cuDoubleComplex, true> {
    static void call(int m, int n, cuDoubleComplex alpha,
                     const cuDoubleComplex *x, const cuDoubleComplex *y,
                     cuDoubleComplex *A, int lda) {
        cublasZgerc(m, n, alpha, x, 1, y, 1, A, lda);
    }
};

template <typename T> static T *legacy_alloc(int n) {
    void *ptr = nullptr;
    CHECK_CUBLAS(cublasAlloc(n, sizeof(T), &ptr));
    return static_cast<T *>(ptr);
}

template <typename T> static void set_vec(const std::vector<T> &host, T *device) {
    CHECK_CUBLAS(cublasSetVector(static_cast<int>(host.size()), sizeof(T),
                                 host.data(), 1, device, 1));
}

template <typename T> static std::vector<T> get_vec(T *device, int n) {
    std::vector<T> out(static_cast<size_t>(n));
    CHECK_CUBLAS(cublasGetVector(n, sizeof(T), device, 1, out.data(), 1));
    return out;
}

template <typename T>
static void expect_vec(const std::vector<T> &got, const std::vector<T> &expected,
                       const char *label) {
    for (size_t i = 0; i < expected.size(); ++i) {
        if (!Ops<T>::near(got[i], expected[i])) {
            std::fprintf(stderr, "%s mismatch at %zu\n", label, i);
            std::exit(1);
        }
    }
}

template <typename T>
static std::vector<T> expected_gemv(const std::vector<T> &A,
                                    const std::vector<T> &x,
                                    const std::vector<T> &y, int m, int n,
                                    T alpha, T beta) {
    std::vector<T> out(static_cast<size_t>(m), Ops<T>::zero());
    for (int row = 0; row < m; ++row) {
        T sum = Ops<T>::zero();
        for (int col = 0; col < n; ++col) {
            sum = Ops<T>::add(sum, Ops<T>::mul(A[row + col * m], x[col]));
        }
        out[row] = Ops<T>::add(Ops<T>::mul(alpha, sum), Ops<T>::mul(beta, y[row]));
    }
    return out;
}

template <typename T>
static std::vector<T> expected_ger(const std::vector<T> &A,
                                   const std::vector<T> &x,
                                   const std::vector<T> &y, int m, int n,
                                   T alpha, bool conjugate) {
    std::vector<T> out = A;
    for (int col = 0; col < n; ++col) {
        T y_value = conjugate ? Ops<T>::conj(y[col]) : y[col];
        for (int row = 0; row < m; ++row) {
            out[row + col * m] = Ops<T>::add(
                out[row + col * m],
                Ops<T>::mul(alpha, Ops<T>::mul(x[row], y_value)));
        }
    }
    return out;
}

template <typename T> static void fill_gemv_inputs(std::vector<T> &A,
                                                   std::vector<T> &x,
                                                   std::vector<T> &y) {
    for (size_t i = 0; i < A.size(); ++i) {
        A[i] = Ops<T>::value(static_cast<float>(i + 1),
                             static_cast<float>((i % 3) - 1) * 0.25f);
    }
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = Ops<T>::value(static_cast<float>(i + 2),
                             static_cast<float>(i + 1) * -0.125f);
    }
    for (size_t i = 0; i < y.size(); ++i) {
        y[i] = Ops<T>::value(static_cast<float>(10 + i),
                             static_cast<float>(i + 1) * 0.375f);
    }
}

template <typename T> static void run_gemv_case(const char *label) {
    const int m = 2;
    const int n = 3;
    std::vector<T> A(static_cast<size_t>(m * n));
    std::vector<T> x(static_cast<size_t>(n));
    std::vector<T> y(static_cast<size_t>(m));
    fill_gemv_inputs(A, x, y);
    T alpha = Ops<T>::alpha();
    T beta = Ops<T>::beta();
    std::vector<T> expected = expected_gemv(A, x, y, m, n, alpha, beta);

    T *d_A = legacy_alloc<T>(m * n);
    T *d_x = legacy_alloc<T>(n);
    T *d_y = legacy_alloc<T>(m);
    set_vec(A, d_A);
    set_vec(x, d_x);
    set_vec(y, d_y);

    CHECK_LAST(LegacyGemv<T>::call(m, n, alpha, d_A, m, d_x, beta, d_y));
    expect_vec(get_vec(d_y, m), expected, label);

    CHECK_CUBLAS(cublasFree(d_A));
    CHECK_CUBLAS(cublasFree(d_x));
    CHECK_CUBLAS(cublasFree(d_y));
}

template <typename T, bool Conjugate>
static void run_ger_case(const char *label) {
    const int m = 2;
    const int n = 3;
    std::vector<T> A(static_cast<size_t>(m * n));
    std::vector<T> x(static_cast<size_t>(m));
    std::vector<T> y(static_cast<size_t>(n));
    for (size_t i = 0; i < A.size(); ++i) {
        A[i] = Ops<T>::value(static_cast<float>(i + 1),
                             static_cast<float>(i) * 0.125f);
    }
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = Ops<T>::value(static_cast<float>(i + 1),
                             static_cast<float>(i + 2) * -0.25f);
    }
    for (size_t i = 0; i < y.size(); ++i) {
        y[i] = Ops<T>::value(static_cast<float>(i + 3),
                             static_cast<float>(i + 1) * 0.5f);
    }
    T alpha = Ops<T>::alpha();
    std::vector<T> expected = expected_ger(A, x, y, m, n, alpha, Conjugate);

    T *d_A = legacy_alloc<T>(m * n);
    T *d_x = legacy_alloc<T>(m);
    T *d_y = legacy_alloc<T>(n);
    set_vec(A, d_A);
    set_vec(x, d_x);
    set_vec(y, d_y);

    CHECK_LAST((LegacyGer<T, Conjugate>::call(m, n, alpha, d_x, d_y, d_A, m)));
    expect_vec(get_vec(d_A, m * n), expected, label);

    CHECK_CUBLAS(cublasFree(d_A));
    CHECK_CUBLAS(cublasFree(d_x));
    CHECK_CUBLAS(cublasFree(d_y));
}

int main() {
    CHECK_CUBLAS(cublasInit());

    run_gemv_case<float>("cublasSgemv");
    run_gemv_case<double>("cublasDgemv");
    run_gemv_case<cuComplex>("cublasCgemv");
    run_gemv_case<cuDoubleComplex>("cublasZgemv");

    run_ger_case<float, false>("cublasSger");
    run_ger_case<double, false>("cublasDger");
    run_ger_case<cuComplex, false>("cublasCgeru");
    run_ger_case<cuComplex, true>("cublasCgerc");
    run_ger_case<cuDoubleComplex, false>("cublasZgeru");
    run_ger_case<cuDoubleComplex, true>("cublasZgerc");

    CHECK_CUBLAS(cublasShutdown());
    std::puts("legacy cuBLAS GEMV/GER API test passed");
    return 0;
}
