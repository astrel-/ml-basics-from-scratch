#include "Matrix.h"
#include <stdexcept>
#include <utility>

namespace matrix {

	Matrix2D::Matrix2D(std::vector<double> buffer_, size_t rows_, size_t cols_)
		: buffer(std::move(buffer_)), rows(rows_), cols(cols_) {
		if (buffer.size() != rows * cols)
			throw std::runtime_error("Underlying vector is incompatible with rows and cols");
	}

	const double* Matrix2D::data() const {
		return buffer.data();
	}

	double& Matrix2D::operator() (size_t row, size_t col) {
		return buffer[row * cols + col];
	}

	const double& Matrix2D::operator() (size_t row, size_t col) const {
		return buffer[row * cols + col];
	}
}