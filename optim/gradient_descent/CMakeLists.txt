add_library(gradient_descent
    src/gradient_descent.cpp
 "src/main.cpp")

target_include_directories(gradient_descent PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(gd_sandbox src/main.cpp)
target_link_libraries(gd_sandbox PRIVATE gradient_descent)