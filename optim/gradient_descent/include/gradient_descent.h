#pragma once
#include <functional>

namespace optim {

enum class ReasonToStop { Iter, Delta, Eps };

struct Solution {
    double x;
    double f;
    int nIter;
    double x_delta;
    double f_delta;
    ReasonToStop reason;
};

class GradientDescentOptimiser {
  public:
    Solution optimise(const std::function<double(double)> &f, const double x0, const double eta,
                    const double eps, const double delta, const int maxIter);
    Solution optimise(const std::function<double(double)> &f,
                    const std::function<double(double)> &grad, const double x0, const double eta,
                    const double eps, const double delta, const int maxIter);
};

} // namespace optim
