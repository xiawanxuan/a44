cmake_minimum_required(VERSION 3.16)

set(TEST_SOURCES
    test_main.cpp
    test_mesh.cpp
    test_sparse.cpp
    test_solver.cpp
    test_assembly.cpp
    test_io.cpp
    test_parallel.cpp
)

add_executable(fem_tests ${TEST_SOURCES})
target_link_libraries(fem_tests PRIVATE fem_core)

add_test(NAME fem_tests COMMAND fem_tests)
