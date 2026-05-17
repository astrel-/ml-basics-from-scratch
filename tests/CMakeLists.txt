add_executable(container_tests
    containers/test_array.cpp
    containers/test_matrix.cpp)

target_link_libraries(container_tests PRIVATE
    GTest::gtest_main
)

target_include_directories(container_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_SOURCE_DIR}/include
)

include(GoogleTest)
gtest_discover_tests(container_tests)