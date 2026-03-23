#include "linalg.h"
#include "matrix.h"
#include <cblas.h>
#include <limits>
#include <stdexcept>
#include <vector>
#include <utility>

namespace linalg {

    [[maybe_unused]] static std::vector<double> matmul_AB_raw(
        const double* A,
        const double* B,
        int k,
        int m,
        int n,
        double alpha = 1.0
    ) {
        // alpha x A[k,m] x B[m,n] = C[k,n]
        std::vector<double> C(static_cast<size_t>(k) * n, 0.0);

        cblas_dgemm(
            CblasRowMajor,
            CblasNoTrans,   // A
            CblasNoTrans,   // B
            k,              // M = rows of C
            n,              // N = cols of C
            m,              // K
            alpha,          // alpha
            A, m,           // lda = number of columns of A
            B, n,           // ldb = number of columns of B
            0.0,            // beta
            C.data(), n     // ldc = number of columns of C
        );

        return C;
    }

    static std::vector<double> matmul_AB_T_raw(
        const double* A,
        const double* B,
        int k,
        int m,
        int n,
        double alpha = 1.0
    ) {
        // alpha x A[k,m] x B[n,m]^T = C[k,n]
        std::vector<double> C(static_cast<size_t>(k) * n, 0.0);

        cblas_dgemm(
            CblasRowMajor,
            CblasNoTrans,   // A
            CblasTrans,   // B
            k,              // M = rows of C
            n,              // N = cols of C
            m,              // K
            alpha,          // alpha
            A, m,           // lda = number of columns of A
            B, m,           // ldb = number of columns of B
            0.0,            // beta
            C.data(), n     // ldc = number of columns of C
        );

        return C;
    }

    static int size_t_to_int(std::size_t value) {
        if (value > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw std::overflow_error("Matrix dimension exceeds BLAS int limit");
        }
        return static_cast<int>(value);
    }

    matrix::Matrix2D matmul_AB(const matrix::Matrix2D& A, const matrix::Matrix2D& B, double alpha) {
        auto k = size_t_to_int(A.rows);
        auto m = size_t_to_int(A.cols);
        if (m != size_t_to_int(B.rows))
            throw std::runtime_error("Incompatible sizes");
        auto n = size_t_to_int(B.cols);

        auto buffer = matmul_AB_T_raw(A.data(), B.data(), k, m, n, alpha);
        return matrix::Matrix2D(std::move(buffer), k, n);
    }

    matrix::Matrix2D matmul_AB_T(const matrix::Matrix2D& A, const matrix::Matrix2D& B, double alpha) {
        auto k = size_t_to_int(A.rows);
        auto m = size_t_to_int(A.cols);
        if (m != size_t_to_int(B.cols))
            throw std::runtime_error("Incompatible sizes");
        auto n = size_t_to_int(B.rows);

        auto buffer = matmul_AB_T_raw(A.data(), B.data(), k, m, n, alpha);
        return matrix::Matrix2D(std::move(buffer), k, n);
    }
}
