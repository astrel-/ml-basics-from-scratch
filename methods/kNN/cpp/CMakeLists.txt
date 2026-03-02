cmake_minimum_required(VERSION 3.20)
project(knn_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# zlib comes from vcpkg (manifest: vcpkg.json at repo root)
find_package(ZLIB REQUIRED)

# cnpy (vendored in repo)
add_library(cnpy STATIC
    ${CMAKE_CURRENT_LIST_DIR}/../../../third_party/cnpy/cnpy.cpp
)

message(STATUS "CMAKE_SOURCE_DIR = ${CMAKE_SOURCE_DIR}")
message(STATUS "CMAKE_CURRENT_SOURCE_DIR = ${CMAKE_CURRENT_SOURCE_DIR}")

target_include_directories(cnpy PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/../../../third_party/cnpy
)
target_link_libraries(cnpy PUBLIC ZLIB::ZLIB)


add_executable(knn
    src/main.cpp
)

get_filename_component(REPO_ROOT
    "${CMAKE_CURRENT_LIST_DIR}/../../../"
    ABSOLUTE
)
target_compile_definitions(knn PRIVATE
    PROJECT_ROOT="${REPO_ROOT}"
)
target_compile_definitions(knn PRIVATE
    TRAIN_DATA_PATH="${REPO_ROOT}/methods/kNN/data/train_data.npz"
)
target_compile_definitions(knn PRIVATE
    TEST_DATA_PATH="${REPO_ROOT}/methods/kNN/data/test_data.npz"
)

target_include_directories(knn PRIVATE include)
target_link_libraries(knn PRIVATE cnpy)