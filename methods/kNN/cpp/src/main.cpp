#include "cnpy.h"
#include <vector>
#include <iostream>
#include <filesystem>

constexpr const char* kTrainPath = TRAIN_DATA_PATH;
constexpr const char* kTestPath = TEST_DATA_PATH;

int main() {
    std::filesystem::path trainPath{ kTrainPath };
    std::filesystem::path testPath{ kTrainPath };

    auto arr = cnpy::npz_load(trainPath.string(), "X");

    double* X = arr.data<double>();

    std::cout << "shape: ";
    for (auto d : arr.shape) std::cout << d << " ";
    std::cout << "\nfirst: " << p[0] << "\n";
}