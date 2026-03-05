#pragma once
#include "Matrix.h"

namespace linalg {
    matrix::Matrix2D matmul_AB(const matrix::Matrix2D& A, const matrix::Matrix2D& B, double alpha = 1.0);
    matrix::Matrix2D matmul_AB_T(const matrix::Matrix2D& A, const matrix::Matrix2D& B, double alpha = 1.0);
}