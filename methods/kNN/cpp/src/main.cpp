#include "knn.h"
#include <iostream>
#include <io/numpy.h>

int main() {
    auto [xTrain, yTrain] = io::numpy::readXY(TRAIN_DATA_PATH);
    auto [xTest, yTest] = io::numpy::readXY(TEST_DATA_PATH);

    int k = 35;
    kNN::CustomKNeighborsClassifier knnClassifier(k);
    knnClassifier.fit(xTrain, yTrain);

    auto yPredNaive = knnClassifier.predict(xTest, kNN::KNeighborsImplementation::Naive);
    auto yPredVec = knnClassifier.predict(xTest, kNN::KNeighborsImplementation::Vectorized);

    for (const auto& y : yPredNaive) {
        std::cout << static_cast<int>(y) << "\n";
    }

    auto matchScore = kNN::util::calcAccuracyScore(yPredNaive, yPredVec);
    std::cout << "Match Score between two methods: " << matchScore << "\n";

    auto accuracyScore = kNN::util::calcAccuracyScore(yPredNaive, yTest);
    std::cout << "Accuracy Score: " << accuracyScore << "\n";
}