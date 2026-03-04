#include "knn.h"
#include <vector>
#include <iostream>
#include <io/numpy.h>

int main() {
    auto [xTrain, yTrain] = io::numpy::readXY(TRAIN_DATA_PATH);
    auto [xTest, yTest] = io::numpy::readXY(TEST_DATA_PATH);

    int k = 1;
    kNN::CustomKNeighborsClassifier knnClassifier(k);
    knnClassifier.fit(xTrain, yTrain);
    auto yPred = knnClassifier.predict(xTest);

    for (const auto& y : yPred) {
        std::cout << static_cast<int>(y) << "\n";
    }

    auto accuracyScore = kNN::util::calcAccuracyScore(yPred, yTest);

    std::cout << "Accuracy Score: " << accuracyScore << "\n";
}