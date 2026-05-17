#include <containers/array.h>
#include <memory>

namespace functions {

template <std::size_t N>
using Func = std::function<double(Array<N>)>;

template <std::size_t N>
using FuncArray = std::array<std::unique_ptr<Func<N>>, N>;
} // namespace functions
