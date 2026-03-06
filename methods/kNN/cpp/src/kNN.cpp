#include "kNN.h"
#include "Matrix.h"
#include "LinAlg.h"
#include <format>
#include <stdexcept>
#include <span>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <numeric> 

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

		double calcSq(std::span<const double> x) {
			double product = 0.0;
			for (size_t idx = 0; idx < x.size(); idx++) {
				product += x[idx] * x[idx];
			}
			return product;
		}

		double calcAccuracyScore(const matrix::Vector1D& yPred, const matrix::Vector1D& yTest)
		{
			int sum = 0;
			auto n = yPred.size();
			for (size_t i = 0; i < n; i++) {
				sum += (yPred[i] == yTest[i]);
			}
			return 1.0 * sum / n;
		}
	}
}

using DistanceHeap = kNN::util::MaxHeapMaxSize<kNN::util::DistanceIndex, kNN::util::CompareDistanceIndex>;

namespace kNN {

	static std::int8_t classifyAndEmptyHeap(DistanceHeap& heap, const matrix::Vector1D& yTrain, std::vector<int>& classCounter, std::int8_t yMin) {
		while (!heap.empty()) {
			auto [_, index] = heap.top();
			auto yClass = static_cast<int8_t>(yTrain[index]);
			int idx = yClass - yMin;
			size_t counterArrayIdx = static_cast<size_t>(idx);
			classCounter[counterArrayIdx]++;
			heap.pop();
		}

		int maxClassCounter = 0;
		std::int8_t maxClassIndex = 0;
		for (std::int8_t i = 0; i < classCounter.size(); i++) {
			auto& cc = classCounter[i];
			if (cc > maxClassCounter) {
				maxClassCounter = cc;
				maxClassIndex = i;
			}
			cc = 0; // empty vector while traversing
		}

		return maxClassIndex + yMin;
	}

	static std::int8_t classify(const matrix::Vector1D& yTrain, std::span<const size_t> nearestIndices, std::vector<int>& classCounter, std::int8_t yMin) {
		for (const auto& idx : nearestIndices) {
			auto yClass = static_cast<int8_t>(yTrain[idx]);
			int clsIdx = yClass - yMin;
			size_t counterArrayIdx = static_cast<size_t>(clsIdx);
			classCounter[counterArrayIdx]++;
		}

		int maxClassCounter = 0;
		std::int8_t maxClassIndex = 0;
		for (std::int8_t i = 0; i < classCounter.size(); i++) {
			auto& cc = classCounter[i];
			if (cc > maxClassCounter) {
				maxClassCounter = cc;
				maxClassIndex = i;
			}
			cc = 0; // empty vector while traversing
		}

		return maxClassIndex + yMin;
	}

	CustomKNeighborsClassifier::CustomKNeighborsClassifier(int n)
		: n_neighbours(n) { }

	void CustomKNeighborsClassifier::fit(const matrix::Matrix2D& xTrain_, const matrix::Vector1D& yTrain_) {
		xTrain = xTrain_;
		yTrain = yTrain_;
		nTrainSamples = xTrain_.rows;
		nTrainFeatures = xTrain_.cols;
		auto [minIt, maxIt] = std::minmax_element(yTrain_.begin(), yTrain_.end());
		yMax = *maxIt;
		yMin = *minIt;
	}

	matrix::Vector1D CustomKNeighborsClassifier::predictNaive(const matrix::Matrix2D& xTest_) const
	{
		auto nSamples = xTest_.rows;
		auto nFeatures = xTest_.cols;
		if (nFeatures != nTrainFeatures)
			throw std::runtime_error(std::format("Train Data has {} features, Test Data has {} features.", nTrainFeatures, nFeatures));
		if (n_neighbours > yTrain.size())
			throw std::runtime_error(std::format("Train Data has fewer features ({}) than k={}.", nTrainFeatures, n_neighbours));
		matrix::Vector1D yPred (nSamples);
		DistanceHeap heap(n_neighbours);
		std::vector<int> classCounter(yMax - yMin + 1);

		for (size_t i = 0; i < nSamples; i++) {
			auto x_i = xTest_.row(i);

			for (size_t j = 0; j < nTrainSamples; j++) {
				auto distance = kNN::util::calcDistanceSq(x_i, xTrain.row(j));
				heap.push({ distance, j });
			}

			yPred[i] = classifyAndEmptyHeap(heap, yTrain, classCounter, yMin);
		}

		return yPred;
	}

	matrix::Vector1D CustomKNeighborsClassifier::predictVectorized(const matrix::Matrix2D& xTest_) const
	{
		auto nSamples = xTest_.rows;
		auto nFeatures = xTest_.cols;
		if (nFeatures != nTrainFeatures)
			throw std::runtime_error(std::format("Train Data has {} features, Test Data has {} features.", nTrainFeatures, nFeatures));
		if (n_neighbours > yTrain.size())
			throw std::runtime_error(std::format("Train Data has fewer features ({}) than k={}.", nTrainFeatures, n_neighbours));
		matrix::Vector1D yPred(nSamples);

		// Matrix of pairwise distances from XTest to XTrain
		// ||a-b||^2 = A^2 + B^2 - 2*AB^T
		matrix::Matrix2D distances{ linalg::matmul_AB_T(xTest_, xTrain, /*alpha=*/-2.0)};
		for (size_t row_idx = 0; row_idx < xTest_.rows; row_idx++) {
			auto aSq = kNN::util::calcSq(xTest_.row(row_idx));
			distances.add_to_row(row_idx, aSq);
		}
		for (size_t col_idx = 0; col_idx < xTrain.rows; col_idx++) {
			auto bSq = kNN::util::calcSq(xTrain.row(col_idx));
			distances.add_to_col(col_idx, bSq);
		}

		DistanceHeap heap(n_neighbours);
		std::vector<int> classCounter(yMax - yMin + 1);

		for (size_t i = 0; i < nSamples; i++) {
			auto distances_i = distances.row(i);
			for (size_t j = 0; j < nTrainSamples; j++) {
				heap.push({ distances_i[j], j});
			}

			yPred[i] = classifyAndEmptyHeap(heap, yTrain, classCounter, yMin);
		}

		return yPred;
	}

	matrix::Vector1D CustomKNeighborsClassifier::predictVectorizedSort(const matrix::Matrix2D& xTest_) const
	{
		auto nSamples = xTest_.rows;
		auto nFeatures = xTest_.cols;
		if (nFeatures != nTrainFeatures)
			throw std::runtime_error(std::format("Train Data has {} features, Test Data has {} features.", nTrainFeatures, nFeatures));
		if (n_neighbours > yTrain.size()) 
			throw std::runtime_error(std::format("Train Data has fewer features ({}) than k={}.", nTrainFeatures, n_neighbours));

		matrix::Vector1D yPred(nSamples);

		// Matrix of pairwise distances from XTest to XTrain
		// ||a-b||^2 = A^2 + B^2 - 2*AB^T
		matrix::Matrix2D distances{ linalg::matmul_AB_T(xTest_, xTrain, /*alpha=*/-2.0) };
		for (size_t row_idx = 0; row_idx < xTest_.rows; row_idx++) {
			auto aSq = kNN::util::calcSq(xTest_.row(row_idx));
			distances.add_to_row(row_idx, aSq);
		}
		for (size_t col_idx = 0; col_idx < xTrain.rows; col_idx++) {
			auto bSq = kNN::util::calcSq(xTrain.row(col_idx));
			distances.add_to_col(col_idx, bSq);
		}

		std::vector<int> classCounter(yMax - yMin + 1);
		std::vector<size_t> indices(yTrain.size());
		
		for (size_t i = 0; i < nSamples; i++) {
			auto distances_i = distances.row(i);
			std::iota(indices.begin(), indices.end(), 0);
			std::sort(indices.begin(), indices.end(),
				[&distances_i](size_t i1, size_t i2) { return distances_i[i1] < distances_i[i2]; });
			yPred[i] = classify(yTrain, std::span(indices).first(n_neighbours), classCounter, yMin);
		}

		return yPred;
	}

	matrix::Vector1D CustomKNeighborsClassifier::predict(const matrix::Matrix2D& xTest_, KNeighborsImplementation impl) const {
		switch (impl) {
			case KNeighborsImplementation::Naive:
				return predictNaive(xTest_);

			case KNeighborsImplementation::Vectorized:
				return predictVectorized(xTest_);

			case KNeighborsImplementation::VectorizedSort:
				return predictVectorizedSort(xTest_);

			default:
				throw std::runtime_error(std::format("Unknown Implementation Method Requested: {}", static_cast<int>(impl)));
		}
	}
}