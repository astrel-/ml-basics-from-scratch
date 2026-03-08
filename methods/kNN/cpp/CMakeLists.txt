cmake_minimum_required(VERSION 3.20)
project(knn_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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

target_link_libraries(cnpy PUBLIC ZLIB::ZLIB)

add_executable(knn
    src/main.cpp
    src/kNN.cpp
    src/io/numpy.cpp
    src/Matrix.cpp
    src/LinAlg.cpp
)

if (MSVC)
    target_compile_options(knn PRIVATE /W4 /analyze-)
else()
    target_compile_options(knn PRIVATE -Wall -Wextra -Wpedantic)
endif()

target_compile_definitions(knn PRIVATE
    PROJECT_ROOT="${CMAKE_SOURCE_DIR}"
    TRAIN_DATA_PATH="${CMAKE_SOURCE_DIR}/methods/kNN/data/train_data.npz"
    TEST_DATA_PATH="${CMAKE_SOURCE_DIR}/methods/kNN/data/test_data.npz"
)

target_include_directories(knn PRIVATE include)

target_link_libraries(knn PRIVATE
    cnpy
    OpenBLAS::OpenBLAS
)