#pragma once
#include "Matrix.h"
#include <vector>
#include <cstdint>

namespace cnpy {
	struct NpyArray;
}

namespace io::numpy {
	matrix::XY readXY(const char* filePath);
	matrix::Matrix2D read2D(const cnpy::NpyArray& array);
	std::vector<std::int8_t> read1D(const cnpy::NpyArray& array);
}