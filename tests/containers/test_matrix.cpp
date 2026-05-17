#include <containers/matrix.h>
#include <gtest/gtest.h>
#include <vector>

namespace {

TEST(MatrixInit, ZeroSize) {

    containers::matrix m{};

    EXPECT_EQ(m.rows, 0);
    EXPECT_EQ(m.cols, 0);
}

TEST(MatrixInit, NonZeroSize) {

    std::vector<double> buf{1, 2, 3, 4};
    std::vector<double> buf_0(4, 0.0);
    containers::matrix m22(buf, 2, 2);
    EXPECT_EQ(m22.rows, 2);
    EXPECT_EQ(m22.cols, 2);

    containers::matrix m14(buf, 1, 4);
    EXPECT_EQ(m14.rows, 1);
    EXPECT_EQ(m14.cols, 4);

    containers::matrix m41(buf, 4, 1);
    EXPECT_EQ(m41.rows, 4);
    EXPECT_EQ(m41.cols, 1);

    auto *data22 = m22.data();
    auto *data14 = m14.data();
    auto *data41 = m41.data();

    for (int i = 0; i != buf.size(); ++i) {
        const auto num = buf[i];
        EXPECT_DOUBLE_EQ(num, *(data22 + i));
        EXPECT_DOUBLE_EQ(num, *(data14 + i));
        EXPECT_DOUBLE_EQ(num, *(data41 + i));
    }
}

} // namespace