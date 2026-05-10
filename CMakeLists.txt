cmake_minimum_required(VERSION 3.20)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(ml_basics_from_scratch LANGUAGES CXX)

add_subdirectory(methods/knn/cpp)
add_subdirectory(optim/gradient_descent)