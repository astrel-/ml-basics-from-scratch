#pragma once
#include "Matrix.h"
#include <cstddef>

namespace kNN {

	class CustomKNeighborsClassifier {

	public:
		CustomKNeighborsClassifier(int n);
		void fit(const matrix::Matrix2D& xTrain_, const matrix::Vector1D& yTrain_);
		matrix::Vector1D predict(const matrix::Matrix2D& xTest_) const;

	private:
		int n_neighbours;
		matrix::Matrix2D xTrain;
		matrix::Vector1D yTrain;
		int yMax;
		int yMin;
		size_t nTrainSamples;
		size_t nTrainFeatures;
	};
}

namespace kNN {
	namespace util {
		double calcDistanceSq(std::span<const double> x, std::span<const double> other);
		double calcAccuracyScore(const matrix::Vector1D& yPred, const matrix::Vector1D& yTest);
	}
}