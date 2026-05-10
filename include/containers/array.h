#pragma once
#include <array>
#include <cmath>
#include <ranges>

namespace containers {

template <std::size_t N>
class array {

  private:
    std::array<double, N> array_;

  public:
    explicit array(const std::array<double, N> &arr) : array_(arr) {}

    array(std::initializer_list<double> values) : array_{} {
        int index = 0;
        for (const auto x : values | std::views::take(N)) {
            array_[index++] = x;
        }
    }

    std::size_t size() const { return N; }

    void operator+=(const array &other) {
        for (int index = 0; index != N; ++index) {
            array_[index] += other.array_[index];
        }
    }

    void operator-=(const array &other) {
        for (int index = 0; index != N; ++index) {
            array_[index] -= other.array_[index];
        }
    }

    double norm_L2_sq() const {
        double norm_sq = 0.0;
        for (const auto x : array_) {
            norm_sq += x * x;
        }
        return norm_sq;
    }

    double norm_L2() const { return std::sqrt(norm_L2_sq()); }

    double norm_L_inf() const {
        double norm = 0.0;
        for (const auto x : array_) {
            norm = std::max(norm, std::abs(x));
        }
        return norm;
    }
};

template <std::size_t N>
array<N> operator-(const array<N> &arr1, const array<N> &arr2) {
    array<N> temp{arr1};
    temp -= arr2;
    return temp;
}

template <std::size_t N>
array<N> operator+(const array<N> &arr1, const array<N> &arr2) {
    array<N> temp{arr1};
    temp += arr2;
    return temp;
}

} // namespace containers
