#pragma once
#include <containers/matrix.h>

namespace linalg {
containers::matrix matmul_AB(const containers::matrix &A, const containers::matrix &B,
                             double alpha = 1.0);
containers::matrix matmul_AB_T(const containers::matrix &A, const containers::matrix &B,
                               double alpha = 1.0);
} // namespace linalg