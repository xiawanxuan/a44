cmake_minimum_required(VERSION 3.16)
project(FEMSparseSolver VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
elseif(MSVC)
    add_compile_options(/W4)
endif()

option(BUILD_TESTS "Build unit tests" ON)
option(ENABLE_OPENMP "Enable OpenMP parallelization" ON)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

if(ENABLE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        add_definitions(-DUSE_OPENMP)
    endif()
endif()

include_directories(${PROJECT_SOURCE_DIR}/include)

file(GLOB_RECURSE FEM_SOURCES
    ${PROJECT_SOURCE_DIR}/src/*.cpp
)

list(FILTER FEM_SOURCES EXCLUDE REGEX "main\\.cpp$")

add_library(fem_core STATIC ${FEM_SOURCES})
target_link_libraries(fem_core PUBLIC Eigen3::Eigen)

if(OpenMP_CXX_FOUND)
    target_link_libraries(fem_core PUBLIC OpenMP::OpenMP_CXX)
endif()

add_executable(fem_solver ${PROJECT_SOURCE_DIR}/src/main.cpp)
target_link_libraries(fem_solver PRIVATE fem_core)

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
