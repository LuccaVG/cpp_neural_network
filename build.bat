@echo off
echo Building Neural Network Prototype...

REM Clean up old files
del *.o 2>nul
del nn_prototype.exe 2>nul

REM Check if MinGW is in PATH, but handle special characters better
set FOUND_GPP=0
for %%X in (g++.exe) do (set FOUND_GPP=1)

IF %FOUND_GPP% EQU 0 (
    echo Adding MinGW to path...
    set "PATH=%PATH%;C:\msys64\mingw64\bin"
)

REM Compile files
echo Compiling neural_network.cpp...
g++ -std=c++17 -Wall -c neural_network.cpp

echo Compiling neural_layer.cpp...
g++ -std=c++17 -Wall -c neural_layer.cpp

echo Compiling main.cpp...
g++ -std=c++17 -Wall -c main.cpp

REM Link files
echo Linking...
g++ -o nn_prototype main.o neural_network.o neural_layer.o

IF %ERRORLEVEL% NEQ 0 (
    echo Build failed!
) ELSE (
    echo Build successful!
    echo Running neural network prototype...
    nn_prototype
)

pause