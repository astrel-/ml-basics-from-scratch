#include <gtest/gtest.h>
#include <containers/array.h>

namespace {

TEST(ArrayInit, ZeroSize) {

    containers::array<0> arr{};

    EXPECT_EQ(arr.size(), 0);
    EXPECT_DOUBLE_EQ(arr.norm_L2(), 0.0);
    EXPECT_DOUBLE_EQ(arr.norm_L_inf(), 0.0);
}

TEST(ArrayInit, NonZeroSize) {

    containers::array<5> arr{1,2,3,4,5};
    constexpr double norm_L2_sq = 1 + 4 + 9 + 16 + 25; 
    constexpr double norm_L_inf = 5.0; 

    EXPECT_EQ(arr.size(), 5);
    EXPECT_DOUBLE_EQ(arr.norm_L2_sq(), norm_L2_sq);
    EXPECT_DOUBLE_EQ(arr.norm_L_inf(), norm_L_inf);
}

TEST(ArrayMaths, Addition) {

    containers::array<5> arr1{1, 2, 3, 4, 5};
    containers::array<5> arr2{2, 3, 4, 5, 6};
    containers::array<5> arr3 = arr1+arr2;

    for (int i = 0; i != 5; i++) {
        EXPECT_EQ(arr3[i], arr1[i] + arr2[i]);
    }
}

TEST(ArrayMaths, Subtraction) {

    containers::array<5> arr1{1, 2, 3, 4, 5};
    containers::array<5> arr2{2, 3, 4, 5, 6};
    containers::array<5> arr3 = arr2 - arr1;

    for (int i = 0; i != 5; i++) {
        EXPECT_EQ(arr3[i], arr2[i] - arr1[i]);
    }
}
} // namespace