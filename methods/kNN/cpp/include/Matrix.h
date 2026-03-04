#pragma once
#include <cstdint>
#include <vector>

namespace matrix {

    using Vector1D = std::vector<std::int8_t>;

    class Matrix2D {
    public:
        Matrix2D(std::vector<double> buffer_, size_t rows_, size_t cols_);

        double& operator() (size_t row, size_t col);
        const double& operator() (size_t row, size_t col) const;

    public:
        size_t rows;
        size_t cols;
        const double* data() const;

    private:
        std::vector<double> buffer;
    };


    struct XY {
        Matrix2D X;
        Vector1D y;
    };
}