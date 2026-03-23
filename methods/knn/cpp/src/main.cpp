#include "knn.h"
#include <iostream>
#include <io/numpy.h>
#include <chrono>
#include <format>


int main() {
    auto [xTrain, yTrain] = io::numpy::readXY(TRAIN_DATA_PATH);
    auto [xTest, yTest] = io::numpy::readXY(TEST_DATA_PATH);

    int k = 20;
    kNN::CustomKNeighborsClassifier knnClassifier(k);
    knnClassifier.fit(xTrain, yTrain);
#ifdef KNN_USE_TORCH
    auto [xTrainTs, yTrainTs] = io::numpy::tensor::readXY(TRAIN_DATA_PATH);
    auto [xTestTs, yTestTs] = io::numpy::tensor::readXY(TEST_DATA_PATH);

    kNN::TorchKNeighborsClassifier torchKnnClassifier(k);
    torchKnnClassifier.fit(xTrainTs, yTrainTs);
    auto yPredTs = torchKnnClassifier.predict(xTestTs);
    std::cout << yPredTs << std::endl;
#endif

    auto yPredNaive = knnClassifier.predict(xTest, kNN::KNeighborsImplementation::Naive);
    auto yPredVec = knnClassifier.predict(xTest, kNN::KNeighborsImplementation::VectorizedHeap);
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
    std::cout << "Match Score between VectorizedSort and VectorizedPartition methods: " << matchScore3 << "\n";

#ifdef KNN_USE_TORCH
    auto accuracyScore0 = kNN::util::calcAccuracyScore(yPredTs, yTestTs);
    std::cout << "Accuracy Score [LibTorch]: " << accuracyScore0 << "\n";
#endif

    auto accuracyScore = kNN::util::calcAccuracyScore(yPredNaive, yTest);
    std::cout << "Accuracy Score: " << accuracyScore << "\n";

    int nRuns = 50;
    auto t1 = std::chrono::steady_clock::now();
    for (int nRun = 0; nRun < nRuns; nRun++) {
        knnClassifier.predict(xTest, kNN::KNeighborsImplementation::Naive);
    }
    auto t2 = std::chrono::steady_clock::now();
    for (int nRun = 0; nRun < nRuns; nRun++) {
        knnClassifier.predict(xTest, kNN::KNeighborsImplementation::VectorizedHeap);
    }
    auto t3 = std::chrono::steady_clock::now();
    for (int nRun = 0; nRun < nRuns; nRun++) {
        knnClassifier.predict(xTest, kNN::KNeighborsImplementation::VectorizedSort);
    }
    auto t4 = std::chrono::steady_clock::now();
    for (int nRun = 0; nRun < nRuns; nRun++) {
        knnClassifier.predict(xTest, kNN::KNeighborsImplementation::VectorizedPartition);
    }
    auto t5 = std::chrono::steady_clock::now();
#ifdef KNN_USE_TORCH
    for (int nRun = 0; nRun < nRuns; nRun++) {
        torchKnnClassifier.predict(xTestTs);
    }
    auto t6 = std::chrono::steady_clock::now();
#endif

    std::cout << std::format("{:<40} {}\n", "Performance [Naive]:", std::chrono::duration_cast<std::chrono::microseconds>((t2 - t1) / nRuns));
    std::cout << std::format("{:<40} {}\n", "Performance [VectorizedHeap]:", std::chrono::duration_cast<std::chrono::microseconds>((t3 - t2) / nRuns));
    std::cout << std::format("{:<40} {}\n", "Performance [VectorizedSort]:", std::chrono::duration_cast<std::chrono::microseconds>((t4 - t3) / nRuns));
    std::cout << std::format("{:<40} {}\n", "Performance [VectorizedPartition]:", std::chrono::duration_cast<std::chrono::microseconds>((t5 - t4) / nRuns));
#ifdef KNN_USE_TORCH
    std::cout << std::format("{:<40} {}\n", "Performance [LibTorch]:", std::chrono::duration_cast<std::chrono::microseconds>((t6- t5) / nRuns));
#endif
}