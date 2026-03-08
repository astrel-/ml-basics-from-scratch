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
#ifdef KNN_USE_TORCH
#include <torch/torch.h>
#endif

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

		matrix::Matrix2D calcPairwiseDistancesSq(const matrix::Matrix2D& xTest, const matrix::Matrix2D& xTrain) {
			// Matrix of pairwise distances from XTest to XTrain
			// ||a-b||^2 = A^2 + B^2 - 2*AB^T
			matrix::Matrix2D distances = linalg::matmul_AB_T(xTest, xTrain, /*alpha=*/-2.0);
			for (size_t row_idx = 0; row_idx < xTest.rows; row_idx++) {
				auto aSq = calcSq(xTest.row(row_idx));
				distances.add_to_row(row_idx, aSq);
			}
			for (size_t col_idx = 0; col_idx < xTrain.rows; col_idx++) {
				auto bSq = calcSq(xTrain.row(col_idx));
				distances.add_to_col(col_idx, bSq);
			}
			return distances;
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

#ifdef KNN_USE_TORCH
		double calcAccuracyScore(const torch::Tensor& yPred, const torch::Tensor& yTest)
		{
			auto correct = (yPred == yTest).sum().item<int64_t>();
			auto n = yPred.size(0);
			return static_cast<double>(correct) / n;
		}
#endif
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
		nFeatures = xTrain_.cols;
		auto [minIt, maxIt] = std::minmax_element(yTrain_.begin(), yTrain_.end());
		yMax = *maxIt;
		yMin = *minIt;
	}

	void CustomKNeighborsClassifier::validate(const matrix::Matrix2D& xTest_) const {
		if (nTrainSamples == 0) {
			throw std::runtime_error("Fit Method hasn't been called. Please call it first");
		}
		auto nTestFeatures = xTest_.cols;
		if (nFeatures != nTestFeatures)
			throw std::runtime_error(std::format("Train Data has {} features, Test Data has {} features.", nFeatures, nTestFeatures));
		if (n_neighbours > yTrain.size())
			throw std::runtime_error(std::format("Train Data has fewer samples ({}) than k={}.", nTrainSamples, n_neighbours));
	}

	matrix::Vector1D CustomKNeighborsClassifier::predictNaive(const matrix::Matrix2D& xTest_) const
	{
		validate(xTest_);
		auto nSamples = xTest_.rows;
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

	matrix::Vector1D CustomKNeighborsClassifier::predictVectorizedHeap(const matrix::Matrix2D& xTest_) const
	{
		validate(xTest_);
		auto nSamples = xTest_.rows;
		matrix::Vector1D yPred(nSamples);

		auto distances = kNN::util::calcPairwiseDistancesSq(xTest_, xTrain);

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
		validate(xTest_);
		auto nSamples = xTest_.rows;
		matrix::Vector1D yPred(nSamples);

		auto distances = kNN::util::calcPairwiseDistancesSq(xTest_, xTrain);

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

	matrix::Vector1D CustomKNeighborsClassifier::predictVectorizedPartition(const matrix::Matrix2D& xTest_) const
	{
		validate(xTest_);
		auto nSamples = xTest_.rows;
		matrix::Vector1D yPred(nSamples);

		auto distances = kNN::util::calcPairwiseDistancesSq(xTest_, xTrain);

		std::vector<int> classCounter(yMax - yMin + 1);
		std::vector<size_t> indices(yTrain.size());

		for (size_t i = 0; i < nSamples; i++) {
			auto distances_i = distances.row(i);
			std::iota(indices.begin(), indices.end(), 0);
			std::nth_element(indices.begin(), indices.begin() + n_neighbours - 1, indices.end(),
				[&distances_i](size_t i1, size_t i2) { return distances_i[i1] < distances_i[i2]; });
			yPred[i] = classify(yTrain, std::span(indices).first(n_neighbours), classCounter, yMin);
		}

		return yPred;
	}

	matrix::Vector1D CustomKNeighborsClassifier::predict(const matrix::Matrix2D& xTest_, KNeighborsImplementation impl) const {
		switch (impl) {
			case KNeighborsImplementation::Naive:
				return predictNaive(xTest_);

			case KNeighborsImplementation::VectorizedHeap:
				return predictVectorizedHeap(xTest_);

			case KNeighborsImplementation::VectorizedSort:
				return predictVectorizedSort(xTest_);

			case KNeighborsImplementation::VectorizedPartition:
				return predictVectorizedPartition(xTest_);

			default:
				throw std::runtime_error(std::format("Unknown Implementation Method Requested: {}", static_cast<int>(impl)));
		}
	}
}

#ifdef KNN_USE_TORCH
namespace kNN {
		TorchKNeighborsClassifier::TorchKNeighborsClassifier(int n)
			: n_neighbours(n) {
		}

		void TorchKNeighborsClassifier::fit(const torch::Tensor& xTrain_, const torch::Tensor& yTrain_) {
			xTrain = xTrain_;
			yTrain = yTrain_;
			yMax = yTrain.max().item<int64_t>();
			yMin = yTrain.min().item<int64_t>();
		}

		torch::Tensor TorchKNeighborsClassifier::predict(const torch::Tensor& xTest_) {
			// [n_test, n_train]
			auto distances = torch::cdist(xTest_, xTrain, 2.0);

			// smallest k distances along train dimension
			auto topk = torch::topk(distances, n_neighbours, /*dim=*/1, /*largest=*/false, /*sorted=*/false);
			auto knn_idx = std::get<1>(topk);   // [n_test, k]

			// gather labels
			auto neighbour_labels =
				yTrain.index_select(0, knn_idx.reshape({ -1 }))
				.reshape({ xTest_.size(0), n_neighbours })
				.to(torch::kInt64);;   // [n_test, k]

			auto num_classes = yMax - yMin + 1;

			auto votes = torch::zeros(
				{ xTest_.size(0), num_classes },
				torch::TensorOptions().dtype(torch::kInt64));

			votes.scatter_add_(
				1,
				neighbour_labels,
				torch::ones_like(neighbour_labels, torch::TensorOptions().dtype(torch::kInt64)));

			return std::get<1>(votes.max(1));   // [n_test]
		}
}
#endif