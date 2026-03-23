#pragma once
#include <cstdint>
#include <span>
#include <vector>

namespace matrix {

    using Vector1D = std::vector<std::int8_t>;

    class Matrix2D {
    public:
        Matrix2D();
        Matrix2D(size_t rows, size_t cols);
        Matrix2D(std::vector<double> buffer_, size_t rows_, size_t cols_);

        double& operator() (size_t row, size_t col);
        const double& operator() (size_t row, size_t col) const;
        std::span<double> row(size_t row);
        std::span<const double> row(size_t row) const;

        void add_to_row(size_t row_idx, double scalar);
        void add_to_col(size_t col_idx, double scalar);

    private:
        std::vector<double> buffer;

    public:
        size_t rows;
        size_t cols;
        const double* data() const;
        double* data();
    };


    struct XY {
        Matrix2D X;
        Vector1D y;
    };
}