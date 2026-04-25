#include "gradient_descent.h"
#include <functional>
#include <iostream>

class TestSquaredFunctions {
  private:
    // funcs
    static double f1(double x, double a) { return (x - a) * (x - a); }
    static double f2(double x1, double x2, double a1, double a2) { return f1(x1, a1) + f1(x2, a2); }

    // grads
    static double g1(double x, double a) { return 2 * (x - a); }

  public:
    // funcs
    static std::function<double(double)> func1(double a) {
        return [a](double x) { return f1(x, a); };
    };

    static std::function<double(double, double)> func2(double a1, double a2) {
        return [a1, a2](double x1, double x2) { return f2(x1, x2, a1, a2); };
    };

    // grads
    static std::function<double(double)> grad1(double a) {
        return [a](double x) { return g1(x, a); };
    };
};

int main() {
    int a1 = 2.0;
    int a2 = 3.0;
    double x0 = 0.0;
    double eta = 0.01;

    auto f1 = TestSquaredFunctions::func1(a1);
    auto f2 = TestSquaredFunctions::func2(a1, a2);

    auto g1 = TestSquaredFunctions::grad1(a1);

    auto opt_gd = optim::GradientDescentOptimiser();

    auto x_gd = opt_gd.optimise(f1, g1, x0, eta, 1e-12, 1e-7, 10'000);

    std::cout << "x0: " << x_gd.x << "\n"
              << "F(x0): " << x_gd.f << "\n"
              << "nIter: " << x_gd.nIter << "\n"
              << "x_delta: " << x_gd.x_delta << "\n"
              << "f_delta: " << x_gd.f_delta << "\n"
              << "reason: " << static_cast<int>(x_gd.reason) << "\n";
}
