#include "cnpy.h"
#include <vector>
#include <cblas.h>
#include <iostream>
#include <filesystem>

constexpr const char* kTrainPath = TRAIN_DATA_PATH;
constexpr const char* kTestPath = TEST_DATA_PATH;

std::vector<double> xxt_from_X(const double* X, int n, int d) {
    std::vector<double> XXt(static_cast<size_t>(n) * n, 0.0);

    cblas_dgemm(
        CblasRowMajor,
        CblasNoTrans,   // X
        CblasTrans,     // X^T
        n,              // M (rows of C)
        n,              // N (cols of C)
        d,              // K
        1.0,            // alpha
        X, d,           // A, lda = d
        X, d,           // B, ldb = d
        0.0,            // beta
        XXt.data(), n   // C, ldc = n
    );

    return XXt;
}


int main() {
    std::filesystem::path trainPath{ kTrainPath };
    std::filesystem::path testPath{ kTestPath };

    auto arr = cnpy::npz_load(trainPath.string(), "X");

    double* X = arr.data<double>();

    auto xxt = xxt_from_X(X, arr.shape[0], arr.shape[1]);

    std::cout << "\nfirst: " << xxt[0] << "\n";
}