project(matmul_example)

set(SOURCES main.cpp)
add_executable(matmul_example ${SOURCES})
target_link_libraries(matmul_example PRIVATE ttml)
