#pragma once
#include <containers/matrix.h>
#ifdef KNN_USE_TORCH
#include <torch/torch.h>
#endif

namespace cnpy {
struct NpyArray;
}

namespace io::numpy {
containers::XY readXY(const char *filePath);
containers::matrix read2D(const cnpy::NpyArray &array);
containers::Vector1D read1D(const cnpy::NpyArray &array);
} // namespace io::numpy

#ifdef KNN_USE_TORCH
namespace io::numpy::tensor {
struct XY {
    torch::Tensor X;
    torch::Tensor y;
};

XY readXY(const char *filePath);
} // namespace io::numpy::tensor
#endif