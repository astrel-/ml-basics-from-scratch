cmake_minimum_required(VERSION 3.20)

project(ml_basics_from_scratch LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

enable_testing()

find_package(GTest CONFIG REQUIRED)
# zlib comes from vcpkg (manifest: vcpkg.json at repo root)
find_package(ZLIB REQUIRED)
# openblas comes from vcpkg
find_package(OpenBLAS CONFIG REQUIRED)

# cnpy (vendored in repo)
add_library(cnpy STATIC
    ${CMAKE_SOURCE_DIR}/third_party/cnpy/cnpy.cpp
)
target_include_directories(cnpy PUBLIC
    ${CMAKE_SOURCE_DIR}/third_party/cnpy
)
target_link_libraries(cnpy PUBLIC
    ZLIB::ZLIB
)
add_library(ml_core
    src/math/linalg.cpp
    src/io/numpy.cpp
)
target_include_directories(ml_core PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(ml_core PUBLIC
    cnpy
    OpenBLAS::OpenBLAS
)

add_subdirectory(methods/knn/cpp)
#add_subdirectory(optim/gradient_descent)
add_subdirectory(tests)