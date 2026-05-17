#pragma once
#include <functional>
#include <memory>

namespace optim {

enum class ReasonToStop { Iter, Delta, Eps };

struct Solution0 {
    double x;
    double f;
    int nIter;
    double x_delta;
    double f_delta;
    ReasonToStop reason;
};






template <std::size_t N>
struct Solution {
    Array<N> x;
    double f;
    int nIter;
    Array<N> x_delta;
    double x_delta_norm; // abs norm
    double f_delta;
    ReasonToStop reason;
};

template <std::size_t N>
class IGradientGenerator {
  private:
    const size_t dims = N;

  public:
    virtual FuncArray<N> generate() = 0;
};

template <std::size_t N>
class Gradient {
  private:
    const FuncArray<N> funcs;

  public:
    Gradient(FuncArray<N> funcs_) : funcs(funcs_) {}
    Gradient(const IGradientGenerator<N> &generator) : funcs(generator.genrate()) {}
};

class GradientDescentOptimiser {
  public:
    template <std::size_t N>
    Solution<N> optimise(const Func<N> &f, const Array<N> x0, const double eta, const double eps,
                         const double delta_norm, const int maxIter) {
        throw std::exception(
            "Optimisation without passing a gradient function is Not Implemented yet");
    }
    template <std::size_t N>
    Solution<N> optimise(const Func<N> &f, const Gradient<N> &grad, const Array<N> x0,
                         const double eta, const double eps, const double delta,
                         const int maxIter) {
        int nIter = 0;
        auto reason = ReasonToStop::Iter;
        Array<N> x_prev = x0;
        Array<N> x_curr = x0;
        Array<N> x_delta = 0.0;
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
};

} // namespace optim
