cmake_minimum_required(VERSION 3.20)

project(ml_basics_from_scratch LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

enable_testing()

find_package(GTest CONFIG REQUIRED)

add_subdirectory(methods/knn/cpp)
#add_subdirectory(optim/gradient_descent)
add_subdirectory(tests)