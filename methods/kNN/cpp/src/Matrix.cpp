#include "Matrix.h"
#include <stdexcept>
#include <utility>
#include <span>
#include <vector>

namespace matrix {
	Matrix2D::Matrix2D() 
		: Matrix2D(std::vector<double> {}, 0, 0) 
	{}

	Matrix2D::Matrix2D(size_t rows_, size_t cols_)
		: buffer(std::vector<double>(rows_ * cols_)), rows(rows_), cols(cols_) 
	{}

	Matrix2D::Matrix2D(std::vector<double> buffer_, size_t rows_, size_t cols_)
		: buffer(std::move(buffer_)), rows(rows_), cols(cols_) 
	{
		if (buffer.size() != rows * cols)
			throw std::runtime_error("Underlying vector is incompatible with rows and cols");
	}

	const double* Matrix2D::data() const {
		return buffer.data();
	}

	double* Matrix2D::data() {
		return buffer.data();
	}

	double& Matrix2D::operator() (size_t row, size_t col) {
		return buffer[row * cols + col];
	}

	const double& Matrix2D::operator() (size_t row, size_t col) const {
		return buffer[row * cols + col];
	}

	std::span<const double> Matrix2D::row(size_t row) const
	{
		return std::span<const double>(data() + row * cols, cols);
	}

	std::span<double> Matrix2D::row(size_t row)
	{
		return std::span<double>(buffer.data() + row * cols, cols);
	}

	void Matrix2D::add_to_row(size_t row_idx, double scalar) {
		for (auto& v : row(row_idx)) {
			v += scalar;
		}
	}

	void Matrix2D::add_to_col(size_t col_idx, double scalar) {
		for (size_t row_idx = 0; row_idx < rows; row_idx++) {
			(*this)(row_idx, col_idx) += scalar;
		}
	}
}