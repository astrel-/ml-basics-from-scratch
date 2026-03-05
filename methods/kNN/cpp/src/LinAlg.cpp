#include "LinAlg.h"
#include "Matrix.h"
#include <cblas.h>
#include <limits>
#include <stdexcept>
#include <vector>

namespace linalg {

    static std::vector<double> matmul_raw(
        const double* A,
        const double* B,
        int k,
        int m,
        int n
    ) {
        // A[k,m] x B[m,n] = C[k,n]
        std::vector<double> C(static_cast<size_t>(k) * n, 0.0);

        cblas_dgemm(
            CblasRowMajor,
            CblasNoTrans,   // A
            CblasNoTrans,   // B
            k,              // M = rows of C
            n,              // N = cols of C
            m,              // K
            1.0,            // alpha
            A, m,           // lda = number of columns of A
            B, n,           // ldb = number of columns of B
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

    matrix::Matrix2D matmul(const matrix::Matrix2D& A, const matrix::Matrix2D& B) {
        auto k = size_t_to_int(A.rows);
        auto m = size_t_to_int(A.cols);
        if (m != size_t_to_int(B.rows))
            throw std::runtime_error("Incompatible sizes");
        auto n = size_t_to_int(B.cols);

        auto buffer = matmul_raw(A.data(), B.data(), k, m, n);
        return matrix::Matrix2D(buffer, k, n);
    }
}
