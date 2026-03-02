#include "cnpy.h"
#include <vector>
#include <iostream>

int main() {
    std::vector<float> X = { 1,2,3,4,5,6 };   // 2x3
    //cnpy::npz_save("train_data.npz", "X", X.data(), { 2,3 }, "w");

    cnpy::NpyArray arr = cnpy::npz_load("train_data.npz", "X");
    float* p = arr.data<float>();

    std::cout << "shape: ";
    for (auto d : arr.shape) std::cout << d << " ";
    std::cout << "\nfirst: " << p[0] << "\n";
}