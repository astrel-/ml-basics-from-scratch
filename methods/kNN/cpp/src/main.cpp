#include "knn.h"
#include <iostream>
#include <io/numpy.h>

int main() {
    auto [xTrain, yTrain] = io::numpy::readXY(TRAIN_DATA_PATH);
    auto [xTest, yTest] = io::numpy::readXY(TEST_DATA_PATH);

    int k = 20;
    kNN::CustomKNeighborsClassifier knnClassifier(k);
    knnClassifier.fit(xTrain, yTrain);

    auto yPredNaive = knnClassifier.predict(xTest, kNN::KNeighborsImplementation::Naive);
    auto yPredVec = knnClassifier.predict(xTest, kNN::KNeighborsImplementation::Vectorized);
    auto yPredVecSort = knnClassifier.predict(xTest, kNN::KNeighborsImplementation::VectorizedSort);
    auto yPredVecPart = knnClassifier.predict(xTest, kNN::KNeighborsImplementation::VectorizedPartition);

    for (const auto& y : yPredNaive) {
        std::cout << static_cast<int>(y) << "\n";
    }

    auto matchScore = kNN::util::calcAccuracyScore(yPredNaive, yPredVec);
    std::cout << "Match Score between Naive and Vectorized methods: " << matchScore << "\n";

    auto matchScore2 = kNN::util::calcAccuracyScore(yPredVecSort, yPredVec);
    std::cout << "Match Score between Vectorized and VectorizedSort methods: " << matchScore2 << "\n";

    auto matchScore3 = kNN::util::calcAccuracyScore(yPredVecSort, yPredVecPart);
    std::cout << "Match Score between VectorizedSort and VectorizedPartition methods: " << matchScore2 << "\n";

    auto accuracyScore = kNN::util::calcAccuracyScore(yPredNaive, yTest);
    std::cout << "Accuracy Score: " << accuracyScore << "\n";
}