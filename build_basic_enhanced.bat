@echo off
echo Building Basic Enhanced Neural Network...

REM Create output directory if it doesn't exist
if not exist "output" mkdir output

REM Simple build excluding problematic files
g++ -std=c++17 -Wall -Wextra -O2 ^
    src/main.cpp ^
    src/neural_network.cpp ^
    src/layers/dense_layer.cpp ^
    src/optimizers/optimizer.cpp ^
    src/core/loss.cpp ^
    src/core/activation.cpp ^
    src/utils/matrix.cpp ^
    src/utils/random.cpp ^
    -Isrc ^
    -o output/basic_enhanced.exe

if %ERRORLEVEL% neq 0 (
    echo Basic build failed!
    exit /b 1
)

echo Basic enhanced neural network built successfully!

REM Build a simple ML demo that doesn't use problematic layers
g++ -std=c++17 -Wall -Wextra -O2 ^
    src/examples/xor.cpp ^
    src/neural_network.cpp ^
    src/layers/dense_layer.cpp ^
    src/optimizers/optimizer.cpp ^
    src/core/loss.cpp ^
    src/core/activation.cpp ^
    src/utils/matrix.cpp ^
    src/utils/random.cpp ^
    -Isrc ^
    -o output/xor_enhanced.exe

if %ERRORLEVEL% neq 0 (
    echo XOR example build failed!
    exit /b 1
)

echo XOR enhanced example built successfully!

echo.
echo ========================================
echo   BASIC BUILD COMPLETED SUCCESSFULLY!
echo ========================================
echo Available executables:
echo   - output/basic_enhanced.exe (Enhanced main)
echo   - output/xor_enhanced.exe (XOR example)
echo.
echo To run the basic enhanced version:
echo   output\basic_enhanced.exe
echo.
