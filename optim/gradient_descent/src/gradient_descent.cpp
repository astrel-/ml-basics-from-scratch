#include "gradient_descent.h"
#include <functional>

namespace optim {

double GradientDescentOptimiser::optimise(std::function<double(double)> f, double x0, double eps,
                                          double delta) {
    double x = 0.0;
    return x;
}
} // namespace optim
