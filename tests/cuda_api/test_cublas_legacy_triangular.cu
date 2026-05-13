#include <cublas.h>
#include <cuComplex.h>

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
    static float mul(float a, float b) { return a * b; }
    static bool near(float a, float b) { return std::fabs(a - b) < 1.0e-4f; }
};

template <> struct Ops<double> {
    static double value(float real, float) { return static_cast<double>(real); }
    static double zero() { return 0.0; }
    static double mul(double a, double b) { return a * b; }
    static bool near(double a, double b) { return std::fabs(a - b) < 1.0e-9; }
};

template <> struct Ops<cuComplex> {
    static cuComplex value(float real, float imag) {
        return make_cuComplex(real, imag);
    }
    static cuComplex zero() { return make_cuComplex(0.0f, 0.0f); }
    static cuComplex mul(cuComplex a, cuComplex b) { return cuCmulf(a, b); }
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
    static cuDoubleComplex mul(cuDoubleComplex a, cuDoubleComplex b) {
        return cuCmul(a, b);
    }
    static bool near(cuDoubleComplex a, cuDoubleComplex b) {
        return std::fabs(a.x - b.x) < 1.0e-9 &&
               std::fabs(a.y - b.y) < 1.0e-9;
    }
};

template <typename T> struct LegacyTri;

template <> struct LegacyTri<float> {
    static void trmv(int n, const float *A, int lda, float *x) {
        cublasStrmv('u', 'n', 'n', n, A, lda, x, 1);
    }
    static void tbmv(int n, const float *A, int lda, float *x) {
        cublasStbmv('u', 'n', 'n', n, 0, A, lda, x, 1);
    }
    static void tpmv(int n, const float *AP, float *x) {
        cublasStpmv('u', 'n', 'n', n, AP, x, 1);
    }
    static void trsv(int n, const float *A, int lda, float *x) {
        cublasStrsv('u', 'n', 'n', n, A, lda, x, 1);
    }
    static void tbsv(int n, const float *A, int lda, float *x) {
        cublasStbsv('u', 'n', 'n', n, 0, A, lda, x, 1);
    }
    static void tpsv(int n, const float *AP, float *x) {
        cublasStpsv('u', 'n', 'n', n, AP, x, 1);
    }
};

template <> struct LegacyTri<double> {
    static void trmv(int n, const double *A, int lda, double *x) {
        cublasDtrmv('u', 'n', 'n', n, A, lda, x, 1);
    }
    static void tbmv(int n, const double *A, int lda, double *x) {
        cublasDtbmv('u', 'n', 'n', n, 0, A, lda, x, 1);
    }
    static void tpmv(int n, const double *AP, double *x) {
        cublasDtpmv('u', 'n', 'n', n, AP, x, 1);
    }
    static void trsv(int n, const double *A, int lda, double *x) {
        cublasDtrsv('u', 'n', 'n', n, A, lda, x, 1);
    }
    static void tbsv(int n, const double *A, int lda, double *x) {
        cublasDtbsv('u', 'n', 'n', n, 0, A, lda, x, 1);
    }
    static void tpsv(int n, const double *AP, double *x) {
        cublasDtpsv('u', 'n', 'n', n, AP, x, 1);
    }
};

template <> struct LegacyTri<cuComplex> {
    static void trmv(int n, const cuComplex *A, int lda, cuComplex *x) {
        cublasCtrmv('u', 'n', 'n', n, A, lda, x, 1);
    }
    static void tbmv(int n, const cuComplex *A, int lda, cuComplex *x) {
        cublasCtbmv('u', 'n', 'n', n, 0, A, lda, x, 1);
    }
    static void tpmv(int n, const cuComplex *AP, cuComplex *x) {
        cublasCtpmv('u', 'n', 'n', n, AP, x, 1);
    }
    static void trsv(int n, const cuComplex *A, int lda, cuComplex *x) {
        cublasCtrsv('u', 'n', 'n', n, A, lda, x, 1);
    }
    static void tbsv(int n, const cuComplex *A, int lda, cuComplex *x) {
        cublasCtbsv('u', 'n', 'n', n, 0, A, lda, x, 1);
    }
    static void tpsv(int n, const cuComplex *AP, cuComplex *x) {
        cublasCtpsv('u', 'n', 'n', n, AP, x, 1);
    }
};

template <> struct LegacyTri<cuDoubleComplex> {
    static void trmv(int n, const cuDoubleComplex *A, int lda,
                     cuDoubleComplex *x) {
        cublasZtrmv('u', 'n', 'n', n, A, lda, x, 1);
    }
    static void tbmv(int n, const cuDoubleComplex *A, int lda,
                     cuDoubleComplex *x) {
        cublasZtbmv('u', 'n', 'n', n, 0, A, lda, x, 1);
    }
    static void tpmv(int n, const cuDoubleComplex *AP, cuDoubleComplex *x) {
        cublasZtpmv('u', 'n', 'n', n, AP, x, 1);
    }
    static void trsv(int n, const cuDoubleComplex *A, int lda,
                     cuDoubleComplex *x) {
        cublasZtrsv('u', 'n', 'n', n, A, lda, x, 1);
    }
    static void tbsv(int n, const cuDoubleComplex *A, int lda,
                     cuDoubleComplex *x) {
        cublasZtbsv('u', 'n', 'n', n, 0, A, lda, x, 1);
    }
    static void tpsv(int n, const cuDoubleComplex *AP, cuDoubleComplex *x) {
        cublasZtpsv('u', 'n', 'n', n, AP, x, 1);
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
static std::vector<T> diagonal_result(const std::vector<T> &diag,
                                      const std::vector<T> &x) {
    std::vector<T> out(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        out[i] = Ops<T>::mul(diag[i], x[i]);
    }
    return out;
}

template <typename T>
static void fill_inputs(std::vector<T> &diag, std::vector<T> &x) {
    for (size_t i = 0; i < diag.size(); ++i) {
        diag[i] = Ops<T>::value(static_cast<float>(i + 2), 0.0f);
        x[i] = Ops<T>::value(static_cast<float>(i + 1),
                             static_cast<float>(i) * 0.25f);
    }
}

template <typename T>
static std::vector<T> full_upper_diag(const std::vector<T> &diag) {
    const int n = static_cast<int>(diag.size());
    std::vector<T> A(static_cast<size_t>(n * n), Ops<T>::zero());
    for (int i = 0; i < n; ++i) {
        A[i + i * n] = diag[static_cast<size_t>(i)];
    }
    return A;
}

template <typename T>
static std::vector<T> packed_upper_diag(const std::vector<T> &diag) {
    const int n = static_cast<int>(diag.size());
    std::vector<T> AP(static_cast<size_t>(n * (n + 1) / 2), Ops<T>::zero());
    for (int col = 0; col < n; ++col) {
        AP[static_cast<size_t>(col * (col + 1) / 2 + col)] =
            diag[static_cast<size_t>(col)];
    }
    return AP;
}

template <typename T>
static void run_full_case(const char *mv_label, const char *sv_label) {
    const int n = 4;
    std::vector<T> diag(static_cast<size_t>(n));
    std::vector<T> x(static_cast<size_t>(n));
    fill_inputs(diag, x);
    std::vector<T> A = full_upper_diag(diag);
    std::vector<T> ax = diagonal_result(diag, x);

    T *d_A = legacy_alloc<T>(n * n);
    T *d_x = legacy_alloc<T>(n);
    set_vec(A, d_A);
    set_vec(x, d_x);
    CHECK_LAST(LegacyTri<T>::trmv(n, d_A, n, d_x));
    expect_vec(get_vec(d_x, n), ax, mv_label);

    set_vec(ax, d_x);
    CHECK_LAST(LegacyTri<T>::trsv(n, d_A, n, d_x));
    expect_vec(get_vec(d_x, n), x, sv_label);

    CHECK_CUBLAS(cublasFree(d_A));
    CHECK_CUBLAS(cublasFree(d_x));
}

template <typename T>
static void run_packed_case(const char *mv_label, const char *sv_label) {
    const int n = 4;
    std::vector<T> diag(static_cast<size_t>(n));
    std::vector<T> x(static_cast<size_t>(n));
    fill_inputs(diag, x);
    std::vector<T> AP = packed_upper_diag(diag);
    std::vector<T> ax = diagonal_result(diag, x);

    T *d_AP = legacy_alloc<T>(static_cast<int>(AP.size()));
    T *d_x = legacy_alloc<T>(n);
    set_vec(AP, d_AP);
    set_vec(x, d_x);
    CHECK_LAST(LegacyTri<T>::tpmv(n, d_AP, d_x));
    expect_vec(get_vec(d_x, n), ax, mv_label);

    set_vec(ax, d_x);
    CHECK_LAST(LegacyTri<T>::tpsv(n, d_AP, d_x));
    expect_vec(get_vec(d_x, n), x, sv_label);

    CHECK_CUBLAS(cublasFree(d_AP));
    CHECK_CUBLAS(cublasFree(d_x));
}

template <typename T>
static void run_band_case(const char *mv_label, const char *sv_label) {
    const int n = 4;
    std::vector<T> diag(static_cast<size_t>(n));
    std::vector<T> x(static_cast<size_t>(n));
    fill_inputs(diag, x);
    std::vector<T> ax = diagonal_result(diag, x);

    T *d_A = legacy_alloc<T>(n);
    T *d_x = legacy_alloc<T>(n);
    set_vec(diag, d_A);
    set_vec(x, d_x);
    CHECK_LAST(LegacyTri<T>::tbmv(n, d_A, 1, d_x));
    expect_vec(get_vec(d_x, n), ax, mv_label);

    set_vec(ax, d_x);
    CHECK_LAST(LegacyTri<T>::tbsv(n, d_A, 1, d_x));
    expect_vec(get_vec(d_x, n), x, sv_label);

    CHECK_CUBLAS(cublasFree(d_A));
    CHECK_CUBLAS(cublasFree(d_x));
}

template <typename T>
static void run_type(const char *trmv, const char *trsv, const char *tpmv,
                     const char *tpsv, const char *tbmv, const char *tbsv) {
    run_full_case<T>(trmv, trsv);
    run_packed_case<T>(tpmv, tpsv);
    run_band_case<T>(tbmv, tbsv);
}

int main() {
    CHECK_CUBLAS(cublasInit());
    run_type<float>("cublasStrmv", "cublasStrsv", "cublasStpmv", "cublasStpsv",
                    "cublasStbmv", "cublasStbsv");
    run_type<double>("cublasDtrmv", "cublasDtrsv", "cublasDtpmv", "cublasDtpsv",
                     "cublasDtbmv", "cublasDtbsv");
    run_type<cuComplex>("cublasCtrmv", "cublasCtrsv", "cublasCtpmv",
                        "cublasCtpsv", "cublasCtbmv", "cublasCtbsv");
    run_type<cuDoubleComplex>("cublasZtrmv", "cublasZtrsv", "cublasZtpmv",
                              "cublasZtpsv", "cublasZtbmv", "cublasZtbsv");
    CHECK_CUBLAS(cublasShutdown());
    std::puts("legacy cuBLAS triangular API test passed");
    return 0;
}
