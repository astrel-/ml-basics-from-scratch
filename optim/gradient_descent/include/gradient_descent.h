#pragma once
#include <functional>

namespace optim {

class GradientDescentOptimiser {
  public:
    double optimise(std::function<double(double)> f, double x0, double eps, double delta);
};

} // namespace optim
