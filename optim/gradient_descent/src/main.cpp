#include "gradient_descent.h"
#include <functional>
#include <iostream>

class TestSquaredFunctions {
  private:
    static double f1(double x, double a) { return (x - a) * (x - a); }
    static double f2(double x1, double x2, double a1, double a2) { return f1(x1, a1) + f1(x2, a2); }

  public:
    static std::function<double(double)> func1(double a) {
        return [a](double x) { return f1(x, a); };
    };

    static std::function<double(double, double)> func2(double a1, double a2) {
        return [a1, a2](double x1, double x2) { return f2(x1, x2, a1, a2); };
    };
};

int main() {

    auto f1 = TestSquaredFunctions::func1(2.0);
    auto f2 = TestSquaredFunctions::func2(2.0, 3.0);

    auto opt_gd = optim::GradientDescentOptimiser();

    auto x_gd = opt_gd.optimise(f1, 0.0, 1e-8, 1e-8);

    std::cout << x_gd << "\n";
}
