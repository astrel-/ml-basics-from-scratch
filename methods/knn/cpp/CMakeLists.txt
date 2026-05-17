cmake_minimum_required(VERSION 3.20)
project(knn_cpp LANGUAGES CXX)

# optional dependency for libtorch
find_package(Torch QUIET)

add_executable(knn
    src/main.cpp
    src/knn.cpp
    
)

if (MSVC)
    target_compile_options(knn PRIVATE /W4 /analyze-)
else()
    target_compile_options(knn PRIVATE -Wall -Wextra -Wpedantic)
endif()

target_compile_definitions(knn PRIVATE
    PROJECT_ROOT="${CMAKE_SOURCE_DIR}"
    TRAIN_DATA_PATH="${CMAKE_SOURCE_DIR}/methods/knn/data/train_data.npz"
    TEST_DATA_PATH="${CMAKE_SOURCE_DIR}/methods/knn/data/test_data.npz"
)

target_include_directories(knn PRIVATE include)

target_link_libraries(knn PRIVATE
    ml_core
)

if (Torch_FOUND)
    target_include_directories(knn PRIVATE ${TORCH_INCLUDE_DIRS})
    target_link_libraries(knn PRIVATE ${TORCH_LIBRARIES})
    target_compile_definitions(knn PRIVATE KNN_USE_TORCH=1)

    if (TORCH_CXX_FLAGS)
        separate_arguments(TORCH_CXX_FLAGS_LIST NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
        target_compile_options(knn PRIVATE ${TORCH_CXX_FLAGS_LIST})
    endif()
endif()