@echo off
echo ======================================================
echo    C++ Neural Network Library - Clean Build System
echo ======================================================

REM Create output directory
if not exist "output" mkdir output

echo.
echo Building Core Neural Network Library...
echo =======================================

REM Build enhanced ML demo (VERIFIED WORKING)
echo [1/2] Building Enhanced ML Demo...
g++ -std=c++17 -O2 -Wall -Wextra ^
    src/ml_working_demo.cpp ^
    src/neural_network.cpp ^
    src/core/activation.cpp ^
    src/core/loss.cpp ^
    src/layers/dense_layer.cpp ^
    src/optimizers/optimizer.cpp ^
    -I src ^
    -o output/neural_network_enhanced.exe

if %errorlevel% neq 0 (
    echo ERROR: Enhanced build failed!
    exit /b 1
)
echo ✓ Enhanced ML Demo built successfully

REM Build advanced features demo
echo [2/2] Building Advanced Features Demo...
g++ -std=c++17 -O2 -Wall -Wextra ^
    src/enhanced_main.cpp ^
    src/neural_network.cpp ^
    src/core/activation.cpp ^
    src/core/loss.cpp ^
    src/layers/dense_layer.cpp ^
    src/optimizers/optimizer.cpp ^
    -I src ^
    -o output/neural_network_advanced.exe

if %errorlevel% neq 0 (
    echo ERROR: Advanced build failed!
    exit /b 1
)
echo ✓ Advanced Features Demo built successfully

echo.
echo ====================================
echo    BUILD COMPLETED SUCCESSFULLY!
echo ====================================
echo.
echo Available Executables:
echo   output\neural_network_enhanced.exe   - Full ML demo with metrics
echo   output\neural_network_advanced.exe   - Advanced features demo
echo.
echo To run the enhanced ML demo (recommended):
echo   output\neural_network_enhanced.exe
echo.
