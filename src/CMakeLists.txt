cmake_minimum_required(VERSION 3.15)

project(CPPNeuralNetwork 
    VERSION 1.0.0
    DESCRIPTION "Enhanced C++ Neural Network Library with Machine Learning Features"
    LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler-specific options
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/core
    ${CMAKE_CURRENT_SOURCE_DIR}/layers
    ${CMAKE_CURRENT_SOURCE_DIR}/optimizers
    ${CMAKE_CURRENT_SOURCE_DIR}/utils
)

# Core library sources
set(CORE_SOURCES
    core/activation.cpp
    core/loss.cpp
    layers/layer.cpp
    layers/dense_layer.cpp
    layers/dropout_layer.cpp
    layers/batch_norm_layer.cpp
    optimizers/optimizer.cpp
    optimizers/sgd.cpp
    optimizers/adam.cpp
    optimizers/momentum.cpp
    optimizers/rmsprop.cpp
    utils/matrix.cpp
    utils/random.cpp
    neural_network.cpp
)

# Create core library
add_library(neural_network_core STATIC ${CORE_SOURCES})

# Enhanced ML Demo
add_executable(neural_network_enhanced ml_working_demo.cpp)
target_link_libraries(neural_network_enhanced neural_network_core)

# Enhanced Main Demo
add_executable(neural_network_advanced enhanced_main.cpp)
target_link_libraries(neural_network_advanced neural_network_core)

# Examples
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/examples)
    file(GLOB EXAMPLE_SOURCES "examples/*.cpp")
    foreach(EXAMPLE_SOURCE ${EXAMPLE_SOURCES})
        get_filename_component(EXAMPLE_NAME ${EXAMPLE_SOURCE} NAME_WE)
        add_executable(${EXAMPLE_NAME} ${EXAMPLE_SOURCE})
        target_link_libraries(${EXAMPLE_NAME} neural_network_core)
    endforeach()
endif()

# Tests (if available)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test)
    enable_testing()
    file(GLOB TEST_SOURCES "test/*.cpp")
    if(TEST_SOURCES)
        add_executable(run_tests ${TEST_SOURCES})
        target_link_libraries(run_tests neural_network_core)
        add_test(NAME NeuralNetworkTests COMMAND run_tests)
    endif()
endif()

# Installation
install(TARGETS neural_network_core neural_network_enhanced neural_network_advanced
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)

# Print build information
message(STATUS "C++ Neural Network Library Configuration:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Executables:")
message(STATUS "    - neural_network_enhanced (Full ML demo)")
message(STATUS "    - neural_network_advanced (Enhanced features)")