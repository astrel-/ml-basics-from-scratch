#pragma once
#include <containers/matrix.h>
#include <queue>
#include <vector>
#include <cstdint>
#include <span>
#ifdef KNN_USE_TORCH
#include <torch/torch.h>
#endif

namespace kNN {

	enum class KNeighborsImplementation {
		Naive,
		VectorizedHeap,
		VectorizedSort,
		VectorizedPartition,
	};

	class CustomKNeighborsClassifier {

	public:
		CustomKNeighborsClassifier(int n);
          void fit(const containers::matrix &xTrain_, const containers::Vector1D &yTrain_);
          containers::Vector1D predict(const containers::matrix &xTest_,
                        KNeighborsImplementation impl = KNeighborsImplementation::Naive) const;

	private:
		int n_neighbours;
        containers::matrix xTrain{};
		containers::Vector1D yTrain{};
		std::int8_t yMax = 0;
		std::int8_t yMin = 0;
		size_t nTrainSamples = 0;
		size_t nFeatures = 0;

		void validate(const containers::matrix &xTest_) const;
		containers::Vector1D predictNaive(const containers::matrix &xTest_) const;
        containers::Vector1D predictVectorizedHeap(const containers::matrix &xTest_) const;
		containers::Vector1D predictVectorizedSort(const containers::matrix &xTest_) const;
		containers::Vector1D predictVectorizedPartition(const containers::matrix &xTest_) const;
	};
}


#ifdef KNN_USE_TORCH
namespace kNN {
		class TorchKNeighborsClassifier {
		public:
			TorchKNeighborsClassifier(int n);
			void fit(const torch::Tensor& xTrain_, const torch::Tensor& yTrain_);
			torch::Tensor predict(const torch::Tensor& xTest_);
		private:
			int n_neighbours;
			torch::Tensor xTrain;
			torch::Tensor yTrain;
			std::int64_t yMax = 0;
			std::int64_t yMin = 0;
		};
}
#endif

namespace kNN {
	namespace util {
		double calcDistanceSq(std::span<const double> x, std::span<const double> other);
        double calcAccuracyScore(const containers::Vector1D &yPred,
                                 const containers::Vector1D &yTest);
#ifdef KNN_USE_TORCH
		double calcAccuracyScore(const torch::Tensor& yPred, const torch::Tensor& yTest);
#endif

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