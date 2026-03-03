#pragma once
#include <cstddef>

namespace kNN {

	class CustomKNeighborsClassifier {

	public:
		CustomKNeighborsClassifier(int n);
		void fit(const double* x, const double* y, std::size_t n_samples, std::size_t n_features);
		const double* predict(const double* xTest, std::size_t n_features);

	private:
		int n_neighbours;
		const double* xTrain;
		const double* yTrain;
		int yMax;
		int yMin;
	};
}