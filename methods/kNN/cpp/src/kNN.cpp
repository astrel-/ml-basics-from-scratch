#include "kNN.h"

namespace kNN {

	CustomKNeighborsClassifier::CustomKNeighborsClassifier(int n)
		: n_neighbours(n) { }

	void CustomKNeighborsClassifier::fit(const double* x, const double* y, std::size_t n_samples, std::size_t n_features) {
		xTrain = x;
		yTrain = y;
	}

	const double* CustomKNeighborsClassifier::predict(const double* x, std::size_t n_features) {
		return yTrain;
	}
}