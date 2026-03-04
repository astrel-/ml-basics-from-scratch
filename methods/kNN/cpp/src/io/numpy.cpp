#include "io/numpy.h"
#include "Matrix.h"
#include <cnpy.h>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace io::numpy {

    matrix::XY readXY(const char* filePath) {
        auto data = cnpy::npz_load(filePath);
        auto xArr = read2D(data["X"]);
        auto yArr = read1D(data["y"]);
        return { xArr, yArr };
    }

    matrix::Matrix2D read2D(const cnpy::NpyArray& array) {
        // consider avoiding copy later
        if (array.fortran_order)
            throw std::runtime_error("Only C order is supported at the moment");
        if (array.shape.size() != 2)
            throw std::runtime_error("Must be a 2D-array");
        if (array.word_size != sizeof(double))
            throw std::runtime_error("Array does not contain doubles");

        auto buffer = array.as_vec<double>();
        auto rows = array.shape[0];
        auto cols = array.shape[1];
        return matrix::Matrix2D(buffer, rows, cols);
    }

    matrix::Vector1D read1D(const cnpy::NpyArray& array) {
        const auto& shape = array.shape;
        if (shape.size() > 2)
            throw std::runtime_error("Must be either 2D-array or 1D-array");
        else if (shape.size() == 2) {
            if ((shape[0] != 1) && (shape[1] != 1))
                throw std::runtime_error("If 2D-array is supplied, it must be of shape (n,1) or (1,n)");
        }
        if (array.word_size != sizeof(std::int8_t))
            throw std::runtime_error("Array does not contain int8_t");

        return array.as_vec<std::int8_t>();
    }
}