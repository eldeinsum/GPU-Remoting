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
    static float alpha() { return 0.75f; }
    static float beta() { return -0.5f; }
    static float real_alpha() { return 0.625f; }
    static bool near(float a, float b) { return std::fabs(a - b) < 1.0e-4f; }
};

template <> struct Ops<double> {
    using Real = double;
    static double value(float real, float) { return static_cast<double>(real); }
    static double zero() { return 0.0; }
    static double add(double a, double b) { return a + b; }
    static double mul(double a, double b) { return a * b; }
    static double conj(double a) { return a; }
    static double alpha() { return 0.75; }
    static double beta() { return -0.5; }
    static double real_alpha() { return 0.625; }
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
    static cuComplex alpha() { return make_cuComplex(0.75f, -0.25f); }
    static cuComplex beta() { return make_cuComplex(-0.5f, 0.125f); }
    static float real_alpha() { return 0.625f; }
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
    static cuDoubleComplex alpha() { return make_cuDoubleComplex(0.75, -0.25); }
    static cuDoubleComplex beta() { return make_cuDoubleComplex(-0.5, 0.125); }
    static double real_alpha() { return 0.625; }
    static bool near(cuDoubleComplex a, cuDoubleComplex b) {
        return std::fabs(a.x - b.x) < 1.0e-9 &&
               std::fabs(a.y - b.y) < 1.0e-9;
    }
};

template <typename T> struct LegacyLevel2;

template <> struct LegacyLevel2<float> {
    static void gbmv(int m, int n, int kl, int ku, float alpha, const float *A,
                     int lda, const float *x, float beta, float *y) {
        cublasSgbmv('n', m, n, kl, ku, alpha, A, lda, x, 1, beta, y, 1);
    }
    static void symv(int n, float alpha, const float *A, int lda,
                     const float *x, float beta, float *y) {
        cublasSsymv('u', n, alpha, A, lda, x, 1, beta, y, 1);
    }
    static void sbmv(int n, int k, float alpha, const float *A, int lda,
                     const float *x, float beta, float *y) {
        cublasSsbmv('u', n, k, alpha, A, lda, x, 1, beta, y, 1);
    }
    static void spmv(int n, float alpha, const float *AP, const float *x,
                     float beta, float *y) {
        cublasSspmv('u', n, alpha, AP, x, 1, beta, y, 1);
    }
    static void syr(int n, float alpha, const float *x, float *A, int lda) {
        cublasSsyr('u', n, alpha, x, 1, A, lda);
    }
    static void spr(int n, float alpha, const float *x, float *AP) {
        cublasSspr('u', n, alpha, x, 1, AP);
    }
    static void syr2(int n, float alpha, const float *x, const float *y,
                     float *A, int lda) {
        cublasSsyr2('u', n, alpha, x, 1, y, 1, A, lda);
    }
    static void spr2(int n, float alpha, const float *x, const float *y,
                     float *AP) {
        cublasSspr2('u', n, alpha, x, 1, y, 1, AP);
    }
};

template <> struct LegacyLevel2<double> {
    static void gbmv(int m, int n, int kl, int ku, double alpha,
                     const double *A, int lda, const double *x, double beta,
                     double *y) {
        cublasDgbmv('n', m, n, kl, ku, alpha, A, lda, x, 1, beta, y, 1);
    }
    static void symv(int n, double alpha, const double *A, int lda,
                     const double *x, double beta, double *y) {
        cublasDsymv('u', n, alpha, A, lda, x, 1, beta, y, 1);
    }
    static void sbmv(int n, int k, double alpha, const double *A, int lda,
                     const double *x, double beta, double *y) {
        cublasDsbmv('u', n, k, alpha, A, lda, x, 1, beta, y, 1);
    }
    static void spmv(int n, double alpha, const double *AP, const double *x,
                     double beta, double *y) {
        cublasDspmv('u', n, alpha, AP, x, 1, beta, y, 1);
    }
    static void syr(int n, double alpha, const double *x, double *A, int lda) {
        cublasDsyr('u', n, alpha, x, 1, A, lda);
    }
    static void spr(int n, double alpha, const double *x, double *AP) {
        cublasDspr('u', n, alpha, x, 1, AP);
    }
    static void syr2(int n, double alpha, const double *x, const double *y,
                     double *A, int lda) {
        cublasDsyr2('u', n, alpha, x, 1, y, 1, A, lda);
    }
    static void spr2(int n, double alpha, const double *x, const double *y,
                     double *AP) {
        cublasDspr2('u', n, alpha, x, 1, y, 1, AP);
    }
};

template <> struct LegacyLevel2<cuComplex> {
    static void gbmv(int m, int n, int kl, int ku, cuComplex alpha,
                     const cuComplex *A, int lda, const cuComplex *x,
                     cuComplex beta, cuComplex *y) {
        cublasCgbmv('n', m, n, kl, ku, alpha, A, lda, x, 1, beta, y, 1);
    }
    static void symv(int n, cuComplex alpha, const cuComplex *A, int lda,
                     const cuComplex *x, cuComplex beta, cuComplex *y) {
        cublasChemv('u', n, alpha, A, lda, x, 1, beta, y, 1);
    }
    static void sbmv(int n, int k, cuComplex alpha, const cuComplex *A, int lda,
                     const cuComplex *x, cuComplex beta, cuComplex *y) {
        cublasChbmv('u', n, k, alpha, A, lda, x, 1, beta, y, 1);
    }
    static void spmv(int n, cuComplex alpha, const cuComplex *AP,
                     const cuComplex *x, cuComplex beta, cuComplex *y) {
        cublasChpmv('u', n, alpha, AP, x, 1, beta, y, 1);
    }
    static void syr(int n, float alpha, const cuComplex *x, cuComplex *A,
                    int lda) {
        cublasCher('u', n, alpha, x, 1, A, lda);
    }
    static void spr(int n, float alpha, const cuComplex *x, cuComplex *AP) {
        cublasChpr('u', n, alpha, x, 1, AP);
    }
    static void syr2(int n, cuComplex alpha, const cuComplex *x,
                     const cuComplex *y, cuComplex *A, int lda) {
        cublasCher2('u', n, alpha, x, 1, y, 1, A, lda);
    }
    static void spr2(int n, cuComplex alpha, const cuComplex *x,
                     const cuComplex *y, cuComplex *AP) {
        cublasChpr2('u', n, alpha, x, 1, y, 1, AP);
    }
};

template <> struct LegacyLevel2<cuDoubleComplex> {
    static void gbmv(int m, int n, int kl, int ku, cuDoubleComplex alpha,
                     const cuDoubleComplex *A, int lda,
                     const cuDoubleComplex *x, cuDoubleComplex beta,
                     cuDoubleComplex *y) {
        cublasZgbmv('n', m, n, kl, ku, alpha, A, lda, x, 1, beta, y, 1);
    }
    static void symv(int n, cuDoubleComplex alpha, const cuDoubleComplex *A,
                     int lda, const cuDoubleComplex *x, cuDoubleComplex beta,
                     cuDoubleComplex *y) {
        cublasZhemv('u', n, alpha, A, lda, x, 1, beta, y, 1);
    }
    static void sbmv(int n, int k, cuDoubleComplex alpha,
                     const cuDoubleComplex *A, int lda,
                     const cuDoubleComplex *x, cuDoubleComplex beta,
                     cuDoubleComplex *y) {
        cublasZhbmv('u', n, k, alpha, A, lda, x, 1, beta, y, 1);
    }
    static void spmv(int n, cuDoubleComplex alpha, const cuDoubleComplex *AP,
                     const cuDoubleComplex *x, cuDoubleComplex beta,
                     cuDoubleComplex *y) {
        cublasZhpmv('u', n, alpha, AP, x, 1, beta, y, 1);
    }
    static void syr(int n, double alpha, const cuDoubleComplex *x,
                    cuDoubleComplex *A, int lda) {
        cublasZher('u', n, alpha, x, 1, A, lda);
    }
    static void spr(int n, double alpha, const cuDoubleComplex *x,
                    cuDoubleComplex *AP) {
        cublasZhpr('u', n, alpha, x, 1, AP);
    }
    static void syr2(int n, cuDoubleComplex alpha, const cuDoubleComplex *x,
                     const cuDoubleComplex *y, cuDoubleComplex *A, int lda) {
        cublasZher2('u', n, alpha, x, 1, y, 1, A, lda);
    }
    static void spr2(int n, cuDoubleComplex alpha, const cuDoubleComplex *x,
                     const cuDoubleComplex *y, cuDoubleComplex *AP) {
        cublasZhpr2('u', n, alpha, x, 1, y, 1, AP);
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
static std::vector<T> make_vector(int n, float offset, bool real_only) {
    std::vector<T> out(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        float real = offset + static_cast<float>(i + 1);
        float imag = real_only ? 0.0f : 0.125f * static_cast<float>(i + 1);
        out[static_cast<size_t>(i)] = Ops<T>::value(real, imag);
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

template <typename T>
static std::vector<T> make_general_band_full(int m, int n, int kl, int ku) {
    std::vector<T> A(static_cast<size_t>(m * n), Ops<T>::zero());
    for (int col = 0; col < n; ++col) {
        int row_min = col - ku > 0 ? col - ku : 0;
        int row_max = col + kl < m - 1 ? col + kl : m - 1;
        for (int row = row_min; row <= row_max; ++row) {
            A[static_cast<size_t>(row + col * m)] =
                Ops<T>::value(static_cast<float>(row + col + 1),
                              0.05f * static_cast<float>(row - col));
        }
    }
    return A;
}

template <typename T>
static std::vector<T> pack_general_band(const std::vector<T> &full, int m,
                                        int n, int kl, int ku) {
    int lda = kl + ku + 1;
    std::vector<T> band(static_cast<size_t>(lda * n), Ops<T>::zero());
    for (int col = 0; col < n; ++col) {
        int row_min = col - ku > 0 ? col - ku : 0;
        int row_max = col + kl < m - 1 ? col + kl : m - 1;
        for (int row = row_min; row <= row_max; ++row) {
            band[static_cast<size_t>(ku + row - col + col * lda)] =
                full[static_cast<size_t>(row + col * m)];
        }
    }
    return band;
}

template <typename T>
static std::vector<T> pack_upper_band(const std::vector<T> &upper, int n,
                                      int k) {
    int lda = k + 1;
    std::vector<T> band(static_cast<size_t>(lda * n), Ops<T>::zero());
    for (int col = 0; col < n; ++col) {
        int row_min = col - k > 0 ? col - k : 0;
        for (int row = row_min; row <= col; ++row) {
            band[static_cast<size_t>(k + row - col + col * lda)] =
                upper[static_cast<size_t>(row + col * n)];
        }
    }
    return band;
}

static size_t packed_upper_index(int row, int col) {
    return static_cast<size_t>(row + col * (col + 1) / 2);
}

template <typename T>
static std::vector<T> pack_upper(const std::vector<T> &upper, int n) {
    std::vector<T> packed(static_cast<size_t>(n * (n + 1) / 2), Ops<T>::zero());
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row <= col; ++row) {
            packed[packed_upper_index(row, col)] =
                upper[static_cast<size_t>(row + col * n)];
        }
    }
    return packed;
}

template <typename T>
static std::vector<T> expected_general_mv(const std::vector<T> &A,
                                          const std::vector<T> &x,
                                          const std::vector<T> &y, int m, int n,
                                          T alpha, T beta) {
    std::vector<T> out(static_cast<size_t>(m), Ops<T>::zero());
    for (int row = 0; row < m; ++row) {
        T sum = Ops<T>::zero();
        for (int col = 0; col < n; ++col) {
            sum = Ops<T>::add(sum, Ops<T>::mul(A[static_cast<size_t>(row + col * m)],
                                               x[static_cast<size_t>(col)]));
        }
        out[static_cast<size_t>(row)] =
            Ops<T>::add(Ops<T>::mul(alpha, sum),
                        Ops<T>::mul(beta, y[static_cast<size_t>(row)]));
    }
    return out;
}

template <typename T>
static T upper_at(const std::vector<T> &upper, int n, int row, int col) {
    if (row <= col) {
        return upper[static_cast<size_t>(row + col * n)];
    }
    return Ops<T>::conj(upper[static_cast<size_t>(col + row * n)]);
}

template <typename T>
static std::vector<T> expected_upper_mv(const std::vector<T> &upper,
                                        const std::vector<T> &x,
                                        const std::vector<T> &y, int n, T alpha,
                                        T beta) {
    std::vector<T> out(static_cast<size_t>(n), Ops<T>::zero());
    for (int row = 0; row < n; ++row) {
        T sum = Ops<T>::zero();
        for (int col = 0; col < n; ++col) {
            sum = Ops<T>::add(sum, Ops<T>::mul(upper_at(upper, n, row, col),
                                               x[static_cast<size_t>(col)]));
        }
        out[static_cast<size_t>(row)] =
            Ops<T>::add(Ops<T>::mul(alpha, sum),
                        Ops<T>::mul(beta, y[static_cast<size_t>(row)]));
    }
    return out;
}

template <typename T>
static std::vector<T> expected_rank1_full(
    const std::vector<T> &A, const std::vector<T> &x, int n,
    typename Ops<T>::Real alpha) {
    std::vector<T> out = A;
    T typed_alpha = Ops<T>::value(static_cast<float>(alpha), 0.0f);
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row <= col; ++row) {
            out[static_cast<size_t>(row + col * n)] = Ops<T>::add(
                out[static_cast<size_t>(row + col * n)],
                Ops<T>::mul(typed_alpha,
                            Ops<T>::mul(x[static_cast<size_t>(row)],
                                        Ops<T>::conj(x[static_cast<size_t>(col)]))));
        }
    }
    return out;
}

template <typename T>
static std::vector<T> expected_rank2_full(const std::vector<T> &A,
                                          const std::vector<T> &x,
                                          const std::vector<T> &y, int n,
                                          T alpha) {
    std::vector<T> out = A;
    T conj_alpha = Ops<T>::conj(alpha);
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row <= col; ++row) {
            T first = Ops<T>::mul(alpha,
                                  Ops<T>::mul(x[static_cast<size_t>(row)],
                                              Ops<T>::conj(y[static_cast<size_t>(col)])));
            T second = Ops<T>::mul(conj_alpha,
                                   Ops<T>::mul(y[static_cast<size_t>(row)],
                                               Ops<T>::conj(x[static_cast<size_t>(col)])));
            out[static_cast<size_t>(row + col * n)] =
                Ops<T>::add(out[static_cast<size_t>(row + col * n)],
                            Ops<T>::add(first, second));
        }
    }
    return out;
}

template <typename T> static void run_gbmv_case(const char *label) {
    const int m = 4;
    const int n = 4;
    const int kl = 1;
    const int ku = 1;
    const int lda = kl + ku + 1;
    std::vector<T> full = make_general_band_full<T>(m, n, kl, ku);
    std::vector<T> band = pack_general_band(full, m, n, kl, ku);
    std::vector<T> x = make_vector<T>(n, 0.0f, false);
    std::vector<T> y = make_vector<T>(m, 4.0f, false);
    T alpha = Ops<T>::alpha();
    T beta = Ops<T>::beta();
    std::vector<T> expected = expected_general_mv(full, x, y, m, n, alpha, beta);

    T *d_A = legacy_alloc<T>(lda * n);
    T *d_x = legacy_alloc<T>(n);
    T *d_y = legacy_alloc<T>(m);
    set_vec(band, d_A);
    set_vec(x, d_x);
    set_vec(y, d_y);
    CHECK_LAST((LegacyLevel2<T>::gbmv(m, n, kl, ku, alpha, d_A, lda, d_x, beta, d_y)));
    expect_vec(get_vec(d_y, m), expected, label);
    CHECK_CUBLAS(cublasFree(d_A));
    CHECK_CUBLAS(cublasFree(d_x));
    CHECK_CUBLAS(cublasFree(d_y));
}

template <typename T> static void run_symv_case(const char *label) {
    const int n = 4;
    std::vector<T> A = make_upper_matrix<T>(n);
    std::vector<T> x = make_vector<T>(n, 0.0f, false);
    std::vector<T> y = make_vector<T>(n, 4.0f, false);
    T alpha = Ops<T>::alpha();
    T beta = Ops<T>::beta();
    std::vector<T> expected = expected_upper_mv(A, x, y, n, alpha, beta);

    T *d_A = legacy_alloc<T>(n * n);
    T *d_x = legacy_alloc<T>(n);
    T *d_y = legacy_alloc<T>(n);
    set_vec(A, d_A);
    set_vec(x, d_x);
    set_vec(y, d_y);
    CHECK_LAST((LegacyLevel2<T>::symv(n, alpha, d_A, n, d_x, beta, d_y)));
    expect_vec(get_vec(d_y, n), expected, label);
    CHECK_CUBLAS(cublasFree(d_A));
    CHECK_CUBLAS(cublasFree(d_x));
    CHECK_CUBLAS(cublasFree(d_y));
}

template <typename T> static void run_sbmv_case(const char *label) {
    const int n = 4;
    const int k = 1;
    const int lda = k + 1;
    std::vector<T> upper = make_upper_matrix<T>(n);
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row <= col; ++row) {
            if (col - row > k) {
                upper[static_cast<size_t>(row + col * n)] = Ops<T>::zero();
            }
        }
    }
    std::vector<T> band = pack_upper_band(upper, n, k);
    std::vector<T> x = make_vector<T>(n, 0.0f, false);
    std::vector<T> y = make_vector<T>(n, 4.0f, false);
    T alpha = Ops<T>::alpha();
    T beta = Ops<T>::beta();
    std::vector<T> expected = expected_upper_mv(upper, x, y, n, alpha, beta);

    T *d_A = legacy_alloc<T>(lda * n);
    T *d_x = legacy_alloc<T>(n);
    T *d_y = legacy_alloc<T>(n);
    set_vec(band, d_A);
    set_vec(x, d_x);
    set_vec(y, d_y);
    CHECK_LAST((LegacyLevel2<T>::sbmv(n, k, alpha, d_A, lda, d_x, beta, d_y)));
    expect_vec(get_vec(d_y, n), expected, label);
    CHECK_CUBLAS(cublasFree(d_A));
    CHECK_CUBLAS(cublasFree(d_x));
    CHECK_CUBLAS(cublasFree(d_y));
}

template <typename T> static void run_spmv_case(const char *label) {
    const int n = 4;
    std::vector<T> upper = make_upper_matrix<T>(n);
    std::vector<T> packed = pack_upper(upper, n);
    std::vector<T> x = make_vector<T>(n, 0.0f, false);
    std::vector<T> y = make_vector<T>(n, 4.0f, false);
    T alpha = Ops<T>::alpha();
    T beta = Ops<T>::beta();
    std::vector<T> expected = expected_upper_mv(upper, x, y, n, alpha, beta);

    T *d_A = legacy_alloc<T>(static_cast<int>(packed.size()));
    T *d_x = legacy_alloc<T>(n);
    T *d_y = legacy_alloc<T>(n);
    set_vec(packed, d_A);
    set_vec(x, d_x);
    set_vec(y, d_y);
    CHECK_LAST((LegacyLevel2<T>::spmv(n, alpha, d_A, d_x, beta, d_y)));
    expect_vec(get_vec(d_y, n), expected, label);
    CHECK_CUBLAS(cublasFree(d_A));
    CHECK_CUBLAS(cublasFree(d_x));
    CHECK_CUBLAS(cublasFree(d_y));
}

template <typename T>
static void run_rank_cases(const char *syr_label, const char *spr_label,
                           const char *syr2_label, const char *spr2_label) {
    const int n = 4;
    std::vector<T> A = make_upper_matrix<T>(n);
    std::vector<T> AP = pack_upper(A, n);
    std::vector<T> x = make_vector<T>(n, 0.0f, true);
    std::vector<T> y = make_vector<T>(n, 3.0f, true);
    typename Ops<T>::Real rank1_alpha = Ops<T>::real_alpha();
    T rank2_alpha = Ops<T>::alpha();
    std::vector<T> expected_syr = expected_rank1_full(A, x, n, rank1_alpha);
    std::vector<T> expected_spr = pack_upper(expected_syr, n);
    std::vector<T> expected_syr2 = expected_rank2_full(A, x, y, n, rank2_alpha);
    std::vector<T> expected_spr2 = pack_upper(expected_syr2, n);

    T *d_A = legacy_alloc<T>(n * n);
    T *d_AP = legacy_alloc<T>(static_cast<int>(AP.size()));
    T *d_x = legacy_alloc<T>(n);
    T *d_y = legacy_alloc<T>(n);

    set_vec(A, d_A);
    set_vec(x, d_x);
    CHECK_LAST((LegacyLevel2<T>::syr(n, rank1_alpha, d_x, d_A, n)));
    expect_upper_matrix(get_vec(d_A, n * n), expected_syr, n, syr_label);

    set_vec(AP, d_AP);
    CHECK_LAST((LegacyLevel2<T>::spr(n, rank1_alpha, d_x, d_AP)));
    expect_vec(get_vec(d_AP, static_cast<int>(AP.size())), expected_spr, spr_label);

    set_vec(A, d_A);
    set_vec(y, d_y);
    CHECK_LAST((LegacyLevel2<T>::syr2(n, rank2_alpha, d_x, d_y, d_A, n)));
    expect_upper_matrix(get_vec(d_A, n * n), expected_syr2, n, syr2_label);

    set_vec(AP, d_AP);
    CHECK_LAST((LegacyLevel2<T>::spr2(n, rank2_alpha, d_x, d_y, d_AP)));
    expect_vec(get_vec(d_AP, static_cast<int>(AP.size())), expected_spr2, spr2_label);

    CHECK_CUBLAS(cublasFree(d_A));
    CHECK_CUBLAS(cublasFree(d_AP));
    CHECK_CUBLAS(cublasFree(d_x));
    CHECK_CUBLAS(cublasFree(d_y));
}

int main() {
    CHECK_CUBLAS(cublasInit());

    run_gbmv_case<float>("cublasSgbmv");
    run_gbmv_case<double>("cublasDgbmv");
    run_gbmv_case<cuComplex>("cublasCgbmv");
    run_gbmv_case<cuDoubleComplex>("cublasZgbmv");

    run_symv_case<float>("cublasSsymv");
    run_symv_case<double>("cublasDsymv");
    run_symv_case<cuComplex>("cublasChemv");
    run_symv_case<cuDoubleComplex>("cublasZhemv");

    run_sbmv_case<float>("cublasSsbmv");
    run_sbmv_case<double>("cublasDsbmv");
    run_sbmv_case<cuComplex>("cublasChbmv");
    run_sbmv_case<cuDoubleComplex>("cublasZhbmv");

    run_spmv_case<float>("cublasSspmv");
    run_spmv_case<double>("cublasDspmv");
    run_spmv_case<cuComplex>("cublasChpmv");
    run_spmv_case<cuDoubleComplex>("cublasZhpmv");

    run_rank_cases<float>("cublasSsyr", "cublasSspr", "cublasSsyr2", "cublasSspr2");
    run_rank_cases<double>("cublasDsyr", "cublasDspr", "cublasDsyr2", "cublasDspr2");
    run_rank_cases<cuComplex>("cublasCher", "cublasChpr", "cublasCher2", "cublasChpr2");
    run_rank_cases<cuDoubleComplex>("cublasZher", "cublasZhpr", "cublasZher2", "cublasZhpr2");

    CHECK_CUBLAS(cublasShutdown());
    std::puts("legacy cuBLAS symmetric/Hermitian Level-2 API test passed");
    return 0;
}
