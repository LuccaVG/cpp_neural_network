@echo off
echo Building Neural Network Prototype with CURL support...

REM Clean up old files
del *.o
del chatbot.exe 2>nul

REM Check if MinGW is in PATH, if not, use the standard location
where g++ >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo Adding MinGW to path...
    REM Adjust this path if your MSYS2 installation is different
    set PATH=%PATH%;C:\msys64\mingw64\bin
)

REM Check if curl-config is available
where curl-config >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo ERROR: curl-config not found! 
    echo Please install curl with: pacman -S mingw-w64-x86_64-curl
    echo Run this command in the MSYS2 MinGW 64-bit shell
    goto error
)

echo Detecting curl configuration...
FOR /F "tokens=*" %%g IN ('curl-config --cflags') do (SET CURL_CFLAGS=%%g)
FOR /F "tokens=*" %%g IN ('curl-config --libs') do (SET CURL_LIBS=%%g)

echo CURL CFLAGS: %CURL_CFLAGS%
echo CURL LIBS: %CURL_LIBS%

REM Compile source files with curl flags
echo Compiling source files...
g++ -std=c++17 -Wall %CURL_CFLAGS% -c main.cpp
g++ -std=c++17 -Wall %CURL_CFLAGS% -c neural_network.cpp
g++ -std=c++17 -Wall %CURL_CFLAGS% -c neural_layer.cpp
g++ -std=c++17 -Wall %CURL_CFLAGS% -c memory.cpp
g++ -std=c++17 -Wall %CURL_CFLAGS% -c chatbot.cpp

REM Link object files with curl library
echo Linking object files...
g++ -o chatbot main.o neural_network.o neural_layer.o memory.o chatbot.o %CURL_LIBS%

IF %ERRORLEVEL% NEQ 0 (
    goto error
)

echo Build successful!
echo Running chatbot...
chatbot
goto end

:error
echo Compilation failed!
echo.
echo Please verify that:
echo 1. MSYS2 MinGW is properly installed
echo 2. curl is installed with: pacman -S mingw-w64-x86_64-curl
echo 3. You're using the MinGW 64-bit terminal or have added it to PATH
echo.

:end
pause