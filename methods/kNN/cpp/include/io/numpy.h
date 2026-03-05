#pragma once
#include "Matrix.h"

namespace cnpy {
	struct NpyArray;
}

namespace io::numpy {
	matrix::XY readXY(const char* filePath);
	matrix::Matrix2D read2D(const cnpy::NpyArray& array);
	matrix::Vector1D read1D(const cnpy::NpyArray& array);
}