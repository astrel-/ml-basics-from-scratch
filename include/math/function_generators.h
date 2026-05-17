// #include "gradient_descent.h"
// #include <cstdarg>
// #include <functional>
// #include <iostream>
#include <containers/array.h>
#include <math/function.h>

namespace functions {

class ParabolicFunctions {

  private:
    // funcs
    static double f1(double x, double a) { return (x - a) * (x - a); }

    template <std::size_t N>
    static double f(const Array<N> &x, const Array<N> &a) {
        double val = 0.0;
        for (int i = 0; i != N; ++i) {
            val += f1(x[i] - a[i]);
        }
        return val;

        // grads
        static double g1(double x, double a) { return 2 * (x - a); }

      public:
        // funcs
        static std::function<double(double)> func1(double a) {
            return [a](double x) { return f(x - a); };
        };

        template <std::size_t N>
        static std::function<double(Array<N>)> func(const Array<N> &a) {
            return [a](const Array<N> &x) { return f(x, a); }
        }

        // grads
        static std::function<double(double)> grad1(double a) {
            return [a](double x) { return g1(x, a); };
        };
    }
}
} // namespace functions
