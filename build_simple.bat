@echo off
echo Building C++ Neural Network Project...

REM Clean up old files
if exist main.exe del main.exe

REM Check if g++ is available
g++ --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: g++ not found. Please install MinGW or add it to PATH.
    pause
    exit /b 1
)

echo Compiling with g++...

REM Compile the project
g++ -std=c++14 -Wall -I src ^
    src/main.cpp ^
    src/neural_network.cpp ^
    src/core/activation.cpp ^
    src/core/loss.cpp ^
    src/layers/dense_layer.cpp ^
    src/optimizers/optimizer.cpp ^
    -o main.exe

if %errorlevel% neq 0 (
    echo Build failed!
    pause
    exit /b 1
) else (
    echo Build successful!
    echo Running the program...
    echo.
    main.exe
)

pause
