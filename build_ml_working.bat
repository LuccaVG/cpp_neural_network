@echo off
echo Building Enhanced Machine Learning Demo...

:: Create output directory
if not exist "output" mkdir output

:: Build the working ML demo
g++ -std=c++17 -O2 -Wall -Wextra ^
    src/ml_working_demo.cpp ^
    src/neural_network.cpp ^
    src/core/activation.cpp ^
    src/core/loss.cpp ^
    src/layers/dense_layer.cpp ^
    src/optimizers/optimizer.cpp ^
    -I src ^
    -o output/ml_working_demo.exe

if %errorlevel% neq 0 (
    echo Build failed!
    exit /b 1
)

echo Build successful! Output: output/ml_working_demo.exe
echo.
echo Running the enhanced machine learning demo...
echo.

:: Run the demo
output\ml_working_demo.exe

if %errorlevel% neq 0 (
    echo Demo execution failed!
    exit /b 1
)

echo.
echo Enhanced ML demo completed successfully!
