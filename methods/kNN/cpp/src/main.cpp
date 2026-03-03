#include <vector>
#include <iostream>
#include <io/numpy.h>

int main() {
    auto [xTrain, yTrain] = io::numpy::readXY(TRAIN_DATA_PATH);
    auto [xTest, yTest] = io::numpy::readXY(TEST_DATA_PATH);

    for (int i = 0; i < 10; i++) {
        std::cout << static_cast<int>(yTest[i]) << "\n";
    }
    std::cout << "x_test" << "\n";

    for (int i = 0; i < 10; i++) {
        std::cout << xTest(i, 1) << "\n";
    }
}