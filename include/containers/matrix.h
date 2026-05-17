#pragma once
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace containers {

class matrix {

  private:
    std::vector<double> buffer;

  public:
    size_t rows;
    size_t cols;

  public:
    matrix() : matrix(std::vector<double>{}, 0, 0) {}

    matrix(size_t rows_, size_t cols_)
        : buffer(std::vector<double>(rows_ * cols_)), rows(rows_), cols(cols_) {}

    matrix(std::vector<double> buffer_, size_t rows_, size_t cols_)
        : buffer(std::move(buffer_)), rows(rows_), cols(cols_) {
        if (buffer.size() != rows * cols)
            throw std::runtime_error("Underlying vector is incompatible with rows and cols");
    }

    double &operator()(size_t row, size_t col) { return buffer[row * cols + col]; }
    const double &operator()(size_t row, size_t col) const { return buffer[row * cols + col]; }

    const double *data() const { return buffer.data(); }
    double *data() { return buffer.data(); }

    std::span<const double> row(size_t row) const {
        return std::span<const double>(data() + row * cols, cols);
    }

    std::span<double> row(size_t row) {
        return std::span<double>(buffer.data() + row * cols, cols);
    }

    void add_to_row(size_t row_idx, double scalar) {
        for (auto &v : row(row_idx)) {
            v += scalar;
        }
    }

    void add_to_col(size_t col_idx, double scalar) {
        for (size_t row_idx = 0; row_idx < rows; row_idx++) {
            (*this)(row_idx, col_idx) += scalar;
        }
    }
};

using Vector1D = std::vector<std::int8_t>;
struct XY {
    matrix X;
    Vector1D y;
};

} // namespace containers