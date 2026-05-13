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
    using Real = float;
    static float value(float real, float) { return real; }
    static float zero() { return 0.0f; }
    static float add(float a, float b) { return a + b; }
    static float mul(float a, float b) { return a * b; }
    static float conj(float a) { return a; }
    static float scale_real(float a, float b) { return a * b; }
    static float alpha() { return 0.75f; }
    static float beta() { return -0.5f; }
    static float real_alpha() { return 0.625f; }
    static float real_beta() { return -0.25f; }
    static bool near(float a, float b) { return std::fabs(a - b) < 1.0e-4f; }
};

template <> struct Ops<double> {
    using Real = double;
    static double value(float real, float) { return static_cast<double>(real); }
    static double zero() { return 0.0; }
    static double add(double a, double b) { return a + b; }
    static double mul(double a, double b) { return a * b; }
    static double conj(double a) { return a; }
    static double scale_real(double a, double b) { return a * b; }
    static double alpha() { return 0.75; }
    static double beta() { return -0.5; }
    static double real_alpha() { return 0.625; }
    static double real_beta() { return -0.25; }
    static bool near(double a, double b) { return std::fabs(a - b) < 1.0e-9; }
};

template <> struct Ops<cuComplex> {
    using Real = float;
    static cuComplex value(float real, float imag) {
        return make_cuComplex(real, imag);
    }
    static cuComplex zero() { return make_cuComplex(0.0f, 0.0f); }
    static cuComplex add(cuComplex a, cuComplex b) { return cuCaddf(a, b); }
    static cuComplex mul(cuComplex a, cuComplex b) { return cuCmulf(a, b); }
    static cuComplex conj(cuComplex a) { return cuConjf(a); }
    static cuComplex scale_real(cuComplex a, float b) {
        return make_cuComplex(a.x * b, a.y * b);
    }
    static cuComplex alpha() { return make_cuComplex(0.75f, -0.25f); }
    static cuComplex beta() { return make_cuComplex(-0.5f, 0.125f); }
    static float real_alpha() { return 0.625f; }
    static float real_beta() { return -0.25f; }
    static bool near(cuComplex a, cuComplex b) {
        return std::fabs(a.x - b.x) < 1.0e-3f &&
               std::fabs(a.y - b.y) < 1.0e-3f;
    }
};

template <> struct Ops<cuDoubleComplex> {
    using Real = double;
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
    static cuDoubleComplex scale_real(cuDoubleComplex a, double b) {
        return make_cuDoubleComplex(a.x * b, a.y * b);
    }
    static cuDoubleComplex alpha() { return make_cuDoubleComplex(0.75, -0.25); }
    static cuDoubleComplex beta() { return make_cuDoubleComplex(-0.5, 0.125); }
    static double real_alpha() { return 0.625; }
    static double real_beta() { return -0.25; }
    static bool near(cuDoubleComplex a, cuDoubleComplex b) {
        return std::fabs(a.x - b.x) < 1.0e-9 &&
               std::fabs(a.y - b.y) < 1.0e-9;
    }
};

template <typename T> struct LegacyL3;

template <> struct LegacyL3<float> {
    static void gemm(int m, int n, int k, float alpha, const float *A, int lda,
                     const float *B, int ldb, float beta, float *C, int ldc) {
        cublasSgemm('n', 'n', m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }
    static void symm(int m, int n, float alpha, const float *A, int lda,
                     const float *B, int ldb, float beta, float *C, int ldc) {
        cublasSsymm('l', 'u', m, n, alpha, A, lda, B, ldb, beta, C, ldc);
    }
    static void syrk(int n, int k, float alpha, const float *A, int lda,
                     float beta, float *C, int ldc) {
        cublasSsyrk('u', 'n', n, k, alpha, A, lda, beta, C, ldc);
    }
    static void syr2k(int n, int k, float alpha, const float *A, int lda,
                      const float *B, int ldb, float beta, float *C, int ldc) {
        cublasSsyr2k('u', 'n', n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }
    static void trmm(int m, int n, float alpha, const float *A, int lda,
                     float *B, int ldb) {
        cublasStrmm('l', 'u', 'n', 'n', m, n, alpha, A, lda, B, ldb);
    }
    static void trsm(int m, int n, float alpha, const float *A, int lda,
                     float *B, int ldb) {
        cublasStrsm('l', 'u', 'n', 'n', m, n, alpha, A, lda, B, ldb);
    }
};

template <> struct LegacyL3<double> {
    static void gemm(int m, int n, int k, double alpha, const double *A, int lda,
                     const double *B, int ldb, double beta, double *C, int ldc) {
        cublasDgemm('n', 'n', m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }
    static void symm(int m, int n, double alpha, const double *A, int lda,
                     const double *B, int ldb, double beta, double *C, int ldc) {
        cublasDsymm('l', 'u', m, n, alpha, A, lda, B, ldb, beta, C, ldc);
    }
    static void syrk(int n, int k, double alpha, const double *A, int lda,
                     double beta, double *C, int ldc) {
        cublasDsyrk('u', 'n', n, k, alpha, A, lda, beta, C, ldc);
    }
    static void syr2k(int n, int k, double alpha, const double *A, int lda,
                      const double *B, int ldb, double beta, double *C,
                      int ldc) {
        cublasDsyr2k('u', 'n', n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }
    static void trmm(int m, int n, double alpha, const double *A, int lda,
                     double *B, int ldb) {
        cublasDtrmm('l', 'u', 'n', 'n', m, n, alpha, A, lda, B, ldb);
    }
    static void trsm(int m, int n, double alpha, const double *A, int lda,
                     double *B, int ldb) {
        cublasDtrsm('l', 'u', 'n', 'n', m, n, alpha, A, lda, B, ldb);
    }
};

template <> struct LegacyL3<cuComplex> {
    static void gemm(int m, int n, int k, cuComplex alpha, const cuComplex *A,
                     int lda, const cuComplex *B, int ldb, cuComplex beta,
                     cuComplex *C, int ldc) {
        cublasCgemm('n', 'n', m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }
    static void symm(int m, int n, cuComplex alpha, const cuComplex *A, int lda,
                     const cuComplex *B, int ldb, cuComplex beta, cuComplex *C,
                     int ldc) {
        cublasCsymm('l', 'u', m, n, alpha, A, lda, B, ldb, beta, C, ldc);
    }
    static void hemm(int m, int n, cuComplex alpha, const cuComplex *A, int lda,
                     const cuComplex *B, int ldb, cuComplex beta, cuComplex *C,
                     int ldc) {
        cublasChemm('l', 'u', m, n, alpha, A, lda, B, ldb, beta, C, ldc);
    }
    static void syrk(int n, int k, cuComplex alpha, const cuComplex *A, int lda,
                     cuComplex beta, cuComplex *C, int ldc) {
        cublasCsyrk('u', 'n', n, k, alpha, A, lda, beta, C, ldc);
    }
    static void herk(int n, int k, float alpha, const cuComplex *A, int lda,
                     float beta, cuComplex *C, int ldc) {
        cublasCherk('u', 'n', n, k, alpha, A, lda, beta, C, ldc);
    }
    static void syr2k(int n, int k, cuComplex alpha, const cuComplex *A, int lda,
                      const cuComplex *B, int ldb, cuComplex beta, cuComplex *C,
                      int ldc) {
        cublasCsyr2k('u', 'n', n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }
    static void her2k(int n, int k, cuComplex alpha, const cuComplex *A, int lda,
                      const cuComplex *B, int ldb, float beta, cuComplex *C,
                      int ldc) {
        cublasCher2k('u', 'n', n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }
    static void trmm(int m, int n, cuComplex alpha, const cuComplex *A, int lda,
                     cuComplex *B, int ldb) {
        cublasCtrmm('l', 'u', 'n', 'n', m, n, alpha, A, lda, B, ldb);
    }
    static void trsm(int m, int n, cuComplex alpha, const cuComplex *A, int lda,
                     cuComplex *B, int ldb) {
        cublasCtrsm('l', 'u', 'n', 'n', m, n, alpha, A, lda, B, ldb);
    }
};

template <> struct LegacyL3<cuDoubleComplex> {
    static void gemm(int m, int n, int k, cuDoubleComplex alpha,
                     const cuDoubleComplex *A, int lda,
                     const cuDoubleComplex *B, int ldb, cuDoubleComplex beta,
                     cuDoubleComplex *C, int ldc) {
        cublasZgemm('n', 'n', m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }
    static void symm(int m, int n, cuDoubleComplex alpha,
                     const cuDoubleComplex *A, int lda,
                     const cuDoubleComplex *B, int ldb, cuDoubleComplex beta,
                     cuDoubleComplex *C, int ldc) {
        cublasZsymm('l', 'u', m, n, alpha, A, lda, B, ldb, beta, C, ldc);
    }
    static void hemm(int m, int n, cuDoubleComplex alpha,
                     const cuDoubleComplex *A, int lda,
                     const cuDoubleComplex *B, int ldb, cuDoubleComplex beta,
                     cuDoubleComplex *C, int ldc) {
        cublasZhemm('l', 'u', m, n, alpha, A, lda, B, ldb, beta, C, ldc);
    }
    static void syrk(int n, int k, cuDoubleComplex alpha,
                     const cuDoubleComplex *A, int lda, cuDoubleComplex beta,
                     cuDoubleComplex *C, int ldc) {
        cublasZsyrk('u', 'n', n, k, alpha, A, lda, beta, C, ldc);
    }
    static void herk(int n, int k, double alpha, const cuDoubleComplex *A,
                     int lda, double beta, cuDoubleComplex *C, int ldc) {
        cublasZherk('u', 'n', n, k, alpha, A, lda, beta, C, ldc);
    }
    static void syr2k(int n, int k, cuDoubleComplex alpha,
                      const cuDoubleComplex *A, int lda,
                      const cuDoubleComplex *B, int ldb, cuDoubleComplex beta,
                      cuDoubleComplex *C, int ldc) {
        cublasZsyr2k('u', 'n', n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }
    static void her2k(int n, int k, cuDoubleComplex alpha,
                      const cuDoubleComplex *A, int lda,
                      const cuDoubleComplex *B, int ldb, double beta,
                      cuDoubleComplex *C, int ldc) {
        cublasZher2k('u', 'n', n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }
    static void trmm(int m, int n, cuDoubleComplex alpha,
                     const cuDoubleComplex *A, int lda, cuDoubleComplex *B,
                     int ldb) {
        cublasZtrmm('l', 'u', 'n', 'n', m, n, alpha, A, lda, B, ldb);
    }
    static void trsm(int m, int n, cuDoubleComplex alpha,
                     const cuDoubleComplex *A, int lda, cuDoubleComplex *B,
                     int ldb) {
        cublasZtrsm('l', 'u', 'n', 'n', m, n, alpha, A, lda, B, ldb);
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
static void expect_upper_matrix(const std::vector<T> &got,
                                const std::vector<T> &expected, int n,
                                const char *label) {
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row <= col; ++row) {
            size_t idx = static_cast<size_t>(row + col * n);
            if (!Ops<T>::near(got[idx], expected[idx])) {
                std::fprintf(stderr, "%s mismatch at (%d,%d)\n", label, row,
                             col);
                std::exit(1);
            }
        }
    }
}

template <typename T>
static std::vector<T> make_matrix(int rows, int cols, float offset,
                                  bool real_only = false) {
    std::vector<T> out(static_cast<size_t>(rows * cols));
    for (int col = 0; col < cols; ++col) {
        for (int row = 0; row < rows; ++row) {
            float real = offset + static_cast<float>(row + 2 * col + 1);
            float imag = real_only ? 0.0f : 0.0625f * static_cast<float>(row - col);
            out[static_cast<size_t>(row + col * rows)] = Ops<T>::value(real, imag);
        }
    }
    return out;
}

template <typename T> static std::vector<T> make_upper_matrix(int n) {
    std::vector<T> A(static_cast<size_t>(n * n), Ops<T>::zero());
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row <= col; ++row) {
            float imag = row == col ? 0.0f : 0.0625f * static_cast<float>(col - row);
            A[static_cast<size_t>(row + col * n)] =
                Ops<T>::value(static_cast<float>(row + 2 * col + 1), imag);
        }
    }
    return A;
}

template <typename T> static std::vector<T> make_triangular_diag(int n) {
    std::vector<T> A(static_cast<size_t>(n * n), Ops<T>::zero());
    for (int i = 0; i < n; ++i) {
        A[static_cast<size_t>(i + i * n)] = Ops<T>::value(static_cast<float>(i + 2), 0.0f);
    }
    return A;
}

template <typename T>
static T upper_at(const std::vector<T> &upper, int n, int row, int col,
                  bool hermitian) {
    if (row <= col) {
        return upper[static_cast<size_t>(row + col * n)];
    }
    T value = upper[static_cast<size_t>(col + row * n)];
    return hermitian ? Ops<T>::conj(value) : value;
}

template <typename T>
static std::vector<T> expected_gemm(const std::vector<T> &A,
                                    const std::vector<T> &B,
                                    const std::vector<T> &C, int m, int n,
                                    int k, T alpha, T beta) {
    std::vector<T> out(static_cast<size_t>(m * n), Ops<T>::zero());
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < m; ++row) {
            T sum = Ops<T>::zero();
            for (int p = 0; p < k; ++p) {
                sum = Ops<T>::add(sum, Ops<T>::mul(A[static_cast<size_t>(row + p * m)],
                                                   B[static_cast<size_t>(p + col * k)]));
            }
            size_t idx = static_cast<size_t>(row + col * m);
            out[idx] = Ops<T>::add(Ops<T>::mul(alpha, sum), Ops<T>::mul(beta, C[idx]));
        }
    }
    return out;
}

template <typename T>
static std::vector<T> expected_symm(const std::vector<T> &A,
                                    const std::vector<T> &B,
                                    const std::vector<T> &C, int m, int n,
                                    T alpha, T beta, bool hermitian) {
    std::vector<T> out(static_cast<size_t>(m * n), Ops<T>::zero());
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < m; ++row) {
            T sum = Ops<T>::zero();
            for (int p = 0; p < m; ++p) {
                sum = Ops<T>::add(sum, Ops<T>::mul(upper_at(A, m, row, p, hermitian),
                                                   B[static_cast<size_t>(p + col * m)]));
            }
            size_t idx = static_cast<size_t>(row + col * m);
            out[idx] = Ops<T>::add(Ops<T>::mul(alpha, sum), Ops<T>::mul(beta, C[idx]));
        }
    }
    return out;
}

template <typename T>
static std::vector<T> expected_syrk(const std::vector<T> &A,
                                    const std::vector<T> &C, int n, int k,
                                    T alpha, T beta, bool hermitian) {
    std::vector<T> out = C;
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row <= col; ++row) {
            T sum = Ops<T>::zero();
            for (int p = 0; p < k; ++p) {
                T right = A[static_cast<size_t>(col + p * n)];
                if (hermitian) {
                    right = Ops<T>::conj(right);
                }
                sum = Ops<T>::add(sum, Ops<T>::mul(A[static_cast<size_t>(row + p * n)], right));
            }
            size_t idx = static_cast<size_t>(row + col * n);
            out[idx] = Ops<T>::add(Ops<T>::mul(alpha, sum), Ops<T>::mul(beta, C[idx]));
        }
    }
    return out;
}

template <typename T>
static std::vector<T> expected_syr2k(const std::vector<T> &A,
                                     const std::vector<T> &B,
                                     const std::vector<T> &C, int n, int k,
                                     T alpha, T beta, bool hermitian) {
    std::vector<T> out = C;
    T conj_alpha = Ops<T>::conj(alpha);
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row <= col; ++row) {
            T sum = Ops<T>::zero();
            for (int p = 0; p < k; ++p) {
                T a_col = A[static_cast<size_t>(col + p * n)];
                T b_col = B[static_cast<size_t>(col + p * n)];
                if (hermitian) {
                    a_col = Ops<T>::conj(a_col);
                    b_col = Ops<T>::conj(b_col);
                }
                T first = Ops<T>::mul(alpha, Ops<T>::mul(A[static_cast<size_t>(row + p * n)], b_col));
                T second_alpha = hermitian ? conj_alpha : alpha;
                T second = Ops<T>::mul(second_alpha,
                                       Ops<T>::mul(B[static_cast<size_t>(row + p * n)], a_col));
                sum = Ops<T>::add(sum, Ops<T>::add(first, second));
            }
            size_t idx = static_cast<size_t>(row + col * n);
            out[idx] = Ops<T>::add(sum, Ops<T>::mul(beta, C[idx]));
        }
    }
    return out;
}

template <typename T>
static std::vector<T> expected_trmm(const std::vector<T> &B, int m, int n,
                                    T alpha, bool solve) {
    std::vector<T> out(static_cast<size_t>(m * n), Ops<T>::zero());
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < m; ++row) {
            typename Ops<T>::Real diag = static_cast<typename Ops<T>::Real>(row + 2);
            typename Ops<T>::Real factor = solve ? (static_cast<typename Ops<T>::Real>(1) / diag) : diag;
            out[static_cast<size_t>(row + col * m)] =
                Ops<T>::scale_real(Ops<T>::mul(alpha, B[static_cast<size_t>(row + col * m)]), factor);
        }
    }
    return out;
}

template <typename T> static void run_gemm_case(const char *label) {
    const int m = 2;
    const int n = 3;
    const int k = 2;
    std::vector<T> A = make_matrix<T>(m, k, 0.0f);
    std::vector<T> B = make_matrix<T>(k, n, 4.0f);
    std::vector<T> C = make_matrix<T>(m, n, 8.0f);
    T alpha = Ops<T>::alpha();
    T beta = Ops<T>::beta();
    std::vector<T> expected = expected_gemm(A, B, C, m, n, k, alpha, beta);

    T *d_A = legacy_alloc<T>(m * k);
    T *d_B = legacy_alloc<T>(k * n);
    T *d_C = legacy_alloc<T>(m * n);
    set_vec(A, d_A);
    set_vec(B, d_B);
    set_vec(C, d_C);
    CHECK_LAST((LegacyL3<T>::gemm(m, n, k, alpha, d_A, m, d_B, k, beta, d_C, m)));
    expect_vec(get_vec(d_C, m * n), expected, label);
    CHECK_CUBLAS(cublasFree(d_A));
    CHECK_CUBLAS(cublasFree(d_B));
    CHECK_CUBLAS(cublasFree(d_C));
}

template <typename T> static void run_symm_case(const char *label) {
    const int m = 3;
    const int n = 2;
    std::vector<T> A = make_upper_matrix<T>(m);
    std::vector<T> B = make_matrix<T>(m, n, 4.0f);
    std::vector<T> C = make_matrix<T>(m, n, 8.0f);
    T alpha = Ops<T>::alpha();
    T beta = Ops<T>::beta();
    std::vector<T> expected = expected_symm(A, B, C, m, n, alpha, beta, false);

    T *d_A = legacy_alloc<T>(m * m);
    T *d_B = legacy_alloc<T>(m * n);
    T *d_C = legacy_alloc<T>(m * n);
    set_vec(A, d_A);
    set_vec(B, d_B);
    set_vec(C, d_C);
    CHECK_LAST((LegacyL3<T>::symm(m, n, alpha, d_A, m, d_B, m, beta, d_C, m)));
    expect_vec(get_vec(d_C, m * n), expected, label);
    CHECK_CUBLAS(cublasFree(d_A));
    CHECK_CUBLAS(cublasFree(d_B));
    CHECK_CUBLAS(cublasFree(d_C));
}

template <typename T> static void run_hemm_case(const char *label) {
    const int m = 3;
    const int n = 2;
    std::vector<T> A = make_upper_matrix<T>(m);
    std::vector<T> B = make_matrix<T>(m, n, 4.0f);
    std::vector<T> C = make_matrix<T>(m, n, 8.0f);
    T alpha = Ops<T>::alpha();
    T beta = Ops<T>::beta();
    std::vector<T> expected = expected_symm(A, B, C, m, n, alpha, beta, true);

    T *d_A = legacy_alloc<T>(m * m);
    T *d_B = legacy_alloc<T>(m * n);
    T *d_C = legacy_alloc<T>(m * n);
    set_vec(A, d_A);
    set_vec(B, d_B);
    set_vec(C, d_C);
    CHECK_LAST((LegacyL3<T>::hemm(m, n, alpha, d_A, m, d_B, m, beta, d_C, m)));
    expect_vec(get_vec(d_C, m * n), expected, label);
    CHECK_CUBLAS(cublasFree(d_A));
    CHECK_CUBLAS(cublasFree(d_B));
    CHECK_CUBLAS(cublasFree(d_C));
}

template <typename T>
static void run_rankk_cases(const char *syrk_label, const char *syr2k_label) {
    const int n = 3;
    const int k = 2;
    std::vector<T> A = make_matrix<T>(n, k, 0.0f);
    std::vector<T> B = make_matrix<T>(n, k, 4.0f);
    std::vector<T> C = make_upper_matrix<T>(n);
    T alpha = Ops<T>::alpha();
    T beta = Ops<T>::beta();
    std::vector<T> expected_syrk_out = expected_syrk(A, C, n, k, alpha, beta, false);
    std::vector<T> expected_syr2k_out = expected_syr2k(A, B, C, n, k, alpha, beta, false);

    T *d_A = legacy_alloc<T>(n * k);
    T *d_B = legacy_alloc<T>(n * k);
    T *d_C = legacy_alloc<T>(n * n);
    set_vec(A, d_A);
    set_vec(C, d_C);
    CHECK_LAST((LegacyL3<T>::syrk(n, k, alpha, d_A, n, beta, d_C, n)));
    expect_upper_matrix(get_vec(d_C, n * n), expected_syrk_out, n, syrk_label);

    set_vec(B, d_B);
    set_vec(C, d_C);
    CHECK_LAST((LegacyL3<T>::syr2k(n, k, alpha, d_A, n, d_B, n, beta, d_C, n)));
    expect_upper_matrix(get_vec(d_C, n * n), expected_syr2k_out, n, syr2k_label);

    CHECK_CUBLAS(cublasFree(d_A));
    CHECK_CUBLAS(cublasFree(d_B));
    CHECK_CUBLAS(cublasFree(d_C));
}

template <typename T>
static void run_herk_cases(const char *herk_label, const char *her2k_label) {
    const int n = 3;
    const int k = 2;
    std::vector<T> A = make_matrix<T>(n, k, 0.0f);
    std::vector<T> B = make_matrix<T>(n, k, 4.0f);
    std::vector<T> C = make_upper_matrix<T>(n);
    typename Ops<T>::Real alpha = Ops<T>::real_alpha();
    typename Ops<T>::Real beta = Ops<T>::real_beta();
    T typed_alpha = Ops<T>::value(static_cast<float>(alpha), 0.0f);
    T typed_beta = Ops<T>::value(static_cast<float>(beta), 0.0f);
    std::vector<T> expected_herk = expected_syrk(A, C, n, k, typed_alpha, typed_beta, true);
    std::vector<T> expected_her2k = expected_syr2k(A, B, C, n, k, Ops<T>::alpha(), typed_beta, true);

    T *d_A = legacy_alloc<T>(n * k);
    T *d_B = legacy_alloc<T>(n * k);
    T *d_C = legacy_alloc<T>(n * n);
    set_vec(A, d_A);
    set_vec(C, d_C);
    CHECK_LAST((LegacyL3<T>::herk(n, k, alpha, d_A, n, beta, d_C, n)));
    expect_upper_matrix(get_vec(d_C, n * n), expected_herk, n, herk_label);

    set_vec(B, d_B);
    set_vec(C, d_C);
    CHECK_LAST((LegacyL3<T>::her2k(n, k, Ops<T>::alpha(), d_A, n, d_B, n, beta, d_C, n)));
    expect_upper_matrix(get_vec(d_C, n * n), expected_her2k, n, her2k_label);

    CHECK_CUBLAS(cublasFree(d_A));
    CHECK_CUBLAS(cublasFree(d_B));
    CHECK_CUBLAS(cublasFree(d_C));
}

template <typename T>
static void run_triangular_cases(const char *trmm_label, const char *trsm_label) {
    const int m = 3;
    const int n = 2;
    std::vector<T> A = make_triangular_diag<T>(m);
    std::vector<T> B = make_matrix<T>(m, n, 4.0f);
    T alpha = Ops<T>::alpha();
    std::vector<T> expected_trmm_out = expected_trmm(B, m, n, alpha, false);
    std::vector<T> expected_trsm_out = expected_trmm(B, m, n, alpha, true);

    T *d_A = legacy_alloc<T>(m * m);
    T *d_B = legacy_alloc<T>(m * n);
    set_vec(A, d_A);
    set_vec(B, d_B);
    CHECK_LAST((LegacyL3<T>::trmm(m, n, alpha, d_A, m, d_B, m)));
    expect_vec(get_vec(d_B, m * n), expected_trmm_out, trmm_label);

    set_vec(B, d_B);
    CHECK_LAST((LegacyL3<T>::trsm(m, n, alpha, d_A, m, d_B, m)));
    expect_vec(get_vec(d_B, m * n), expected_trsm_out, trsm_label);

    CHECK_CUBLAS(cublasFree(d_A));
    CHECK_CUBLAS(cublasFree(d_B));
}

int main() {
    CHECK_CUBLAS(cublasInit());

    run_gemm_case<float>("cublasSgemm");
    run_gemm_case<double>("cublasDgemm");
    run_gemm_case<cuComplex>("cublasCgemm");
    run_gemm_case<cuDoubleComplex>("cublasZgemm");

    run_symm_case<float>("cublasSsymm");
    run_symm_case<double>("cublasDsymm");
    run_symm_case<cuComplex>("cublasCsymm");
    run_symm_case<cuDoubleComplex>("cublasZsymm");
    run_hemm_case<cuComplex>("cublasChemm");
    run_hemm_case<cuDoubleComplex>("cublasZhemm");

    run_rankk_cases<float>("cublasSsyrk", "cublasSsyr2k");
    run_rankk_cases<double>("cublasDsyrk", "cublasDsyr2k");
    run_rankk_cases<cuComplex>("cublasCsyrk", "cublasCsyr2k");
    run_rankk_cases<cuDoubleComplex>("cublasZsyrk", "cublasZsyr2k");
    run_herk_cases<cuComplex>("cublasCherk", "cublasCher2k");
    run_herk_cases<cuDoubleComplex>("cublasZherk", "cublasZher2k");

    run_triangular_cases<float>("cublasStrmm", "cublasStrsm");
    run_triangular_cases<double>("cublasDtrmm", "cublasDtrsm");
    run_triangular_cases<cuComplex>("cublasCtrmm", "cublasCtrsm");
    run_triangular_cases<cuDoubleComplex>("cublasZtrmm", "cublasZtrsm");

    CHECK_CUBLAS(cublasShutdown());
    std::puts("legacy cuBLAS Level-3 API test passed");
    return 0;
}
