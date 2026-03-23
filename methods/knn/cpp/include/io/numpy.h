#pragma once
#include "Matrix.h"
#ifdef KNN_USE_TORCH
#include <torch/torch.h>
#endif

namespace cnpy {
	struct NpyArray;
}

namespace io::numpy {
	matrix::XY readXY(const char* filePath);
	matrix::Matrix2D read2D(const cnpy::NpyArray& array);
	matrix::Vector1D read1D(const cnpy::NpyArray& array);
}

#ifdef KNN_USE_TORCH
namespace io::numpy::tensor {
	struct XY {
		torch::Tensor X;
		torch::Tensor y;
	};

	XY readXY(const char* filePath);
}
#endif