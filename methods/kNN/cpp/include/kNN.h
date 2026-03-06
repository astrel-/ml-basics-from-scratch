#pragma once
#include "Matrix.h"
#include <queue>
#include <vector>
#include <cstdint>
#include <span>

namespace kNN {

	enum class KNeighborsImplementation {
		Naive,
		Vectorized,
		VectorizedSort,
	};

	class CustomKNeighborsClassifier {

	public:
		CustomKNeighborsClassifier(int n);
		void fit(const matrix::Matrix2D& xTrain_, const matrix::Vector1D& yTrain_);
		matrix::Vector1D predict(const matrix::Matrix2D& xTest_, KNeighborsImplementation impl = KNeighborsImplementation::Naive) const;

	private:
		int n_neighbours;
		matrix::Matrix2D xTrain{};
		matrix::Vector1D yTrain{};
		std::int8_t yMax = 0;
		std::int8_t yMin = 0;
		size_t nTrainSamples = 0;
		size_t nFeatures = 0;

		void validate(const matrix::Matrix2D& xTest_) const;
		matrix::Vector1D predictNaive(const matrix::Matrix2D& xTest_) const;
		matrix::Vector1D predictVectorized(const matrix::Matrix2D& xTest_) const;
		matrix::Vector1D predictVectorizedSort(const matrix::Matrix2D& xTest_) const;
	};
}

namespace kNN {
	namespace util {
		double calcDistanceSq(std::span<const double> x, std::span<const double> other);
		double calcAccuracyScore(const matrix::Vector1D& yPred, const matrix::Vector1D& yTest);

		struct DistanceIndex {
			double distance;
			size_t index;
		};

		struct CompareDistanceIndex {
			bool operator()(const DistanceIndex& a, const DistanceIndex& b) const {
				return a.distance < b.distance;
			}
		};

		template<class T, class Compare>
		class MaxHeapMaxSize {
		public:
			MaxHeapMaxSize(size_t maxSize_)
				: maxSize(maxSize_) {}
			void push(const T& value) {
				if (maxHeap.size() < maxSize) {
					maxHeap.push(value);
					return;
				}
				if (comp(value, maxHeap.top())) {
					maxHeap.pop();
					maxHeap.push(value);
					return;
				}
				return;
			}

			const T& top() const {
				return maxHeap.top();
			}

			void pop() {
				maxHeap.pop();
			}

			bool empty() const {
				return maxHeap.empty();
			}

		private:
			size_t maxSize;
			std::priority_queue<T, std::vector<T>, Compare> maxHeap;
			Compare comp;
		};
	}
}