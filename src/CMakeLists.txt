# CMakeLists.txt

cmake_minimum_required(VERSION 3.10)

project(CPPNeuralNetwork)

set(CMAKE_CXX_STANDARD 14)

# Include directories
include_directories(
    src
    src/core
    src/optimizers
    src/layers
    src/utils
)

# Source files
file(GLOB_RECURSE SOURCES
    "src/*.cpp"
)

# Add executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Add test executable
enable_testing()
file(GLOB TEST_SOURCES "test/*.cpp")
add_executable(runTests ${TEST_SOURCES})
target_link_libraries(runTests ${PROJECT_NAME})
add_test(NAME UnitTests COMMAND runTests)

# Add examples
file(GLOB EXAMPLES "examples/*.cpp")
foreach(EXAMPLE ${EXAMPLES})
    get_filename_component(EXAMPLE_NAME ${EXAMPLE} NAME_WE)
    add_executable(${EXAMPLE_NAME} ${EXAMPLE})
    target_link_libraries(${EXAMPLE_NAME} ${PROJECT_NAME})
endforeach()