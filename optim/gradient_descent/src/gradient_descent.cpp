#include "gradient_descent.h"
#include <cstdlib>
#include <exception>
#include <functional>

namespace optim {

Solution GradientDescentOptimiser::optimise(const std::function<double(double)> &f, const double x0,
                                            const double eta, const double eps, const double delta,
                                            const int maxIter) {
    throw std::exception("Optimisation without passing a gradient function is Not Implemented yet");
}

Solution GradientDescentOptimiser::optimise(const std::function<double(double)> &f,
                                            const std::function<double(double)> &grad,
                                            const double x0, const double eta, const double eps,
                                            const double delta, const int maxIter) {
    int nIter = 0;
    auto reason = ReasonToStop::Iter;
    double x_prev = x0;
    double x_curr = x0;
    double x_delta = 0.0;
    double f_prev = f(x_prev);
    double f_curr = 0.0;
    double f_delta = 0.0;
    while (nIter < maxIter) {
        x_curr = x_prev - eta * grad(x_prev);
        f_curr = f(x_curr);

        f_delta = std::abs(f_curr - f_prev);
        x_delta = std::abs(x_curr - x_prev);
        if (f_delta < eps) {
            reason = ReasonToStop::Eps;
            break;
        }

        if (x_delta < delta) {
            reason = ReasonToStop::Delta;
            break;
        }

        x_prev = x_curr;
        f_prev = f_curr;
        nIter++;
    }
    return Solution{x_curr, f_curr, nIter, x_delta, f_delta, reason};
}
} // namespace optim
