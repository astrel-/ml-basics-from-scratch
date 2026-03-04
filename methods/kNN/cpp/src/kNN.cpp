#include "kNN.h"
#include "Matrix.h"
#include <format>
#include <stdexcept>
#include <limits>
#include <span>

namespace kNN {
	namespace util {

		double calcDistanceSq(std::span<const double> x, std::span<const double> other) {
			double product = 0.0;
			for (size_t idx = 0; idx < x.size(); idx++) {
				double diff = x[idx] - other[idx];
				product += diff * diff;
			}
			return product;
		}

		double calcAccuracyScore(const matrix::Vector1D& yPred, const matrix::Vector1D& yTest)
		{
			int sum = 0;
			int n = yPred.size();
			for (size_t i = 0; i < n; i++) {
				sum += (yPred[i] == yTest[i]);
			}
			return 1.0 * sum / n;
		}
	}
}


namespace kNN {

	CustomKNeighborsClassifier::CustomKNeighborsClassifier(int n)
		: n_neighbours(n) { }

	void CustomKNeighborsClassifier::fit(const matrix::Matrix2D& xTrain_, const matrix::Vector1D& yTrain_) {
		xTrain = xTrain_;
		yTrain = yTrain_;
		nTrainSamples = xTrain_.rows;
		nTrainFeatures = xTrain_.cols;
	}

	matrix::Vector1D CustomKNeighborsClassifier::predict(const matrix::Matrix2D& xTest_) const
	{
		matrix::Vector1D yPred;
		auto nSamples = xTest_.rows;
		auto nFeatures = xTest_.cols;
		if (nFeatures != nTrainFeatures)
			throw std::runtime_error(std::format("Train Data has {} features, Test Data has {} features.", nTrainFeatures, nFeatures));
		yPred.resize(nSamples);

		for (int i = 0; i < nSamples; i++) {
			const auto& x_i = xTest_.row(i);
			int closestNeighborIndex = 0;
			double closestNeighbourDistance = std::numeric_limits<double>::infinity();

			for (int j = 0; j < nTrainSamples; j++) {
				auto distance = kNN::util::calcDistanceSq(x_i, xTrain.row(j));
				if (distance < closestNeighbourDistance) {
					closestNeighbourDistance = distance;
					closestNeighborIndex = j;
				}
			}
			yPred[i] = yTrain[closestNeighborIndex];
		}

		return yPred;
	}
}