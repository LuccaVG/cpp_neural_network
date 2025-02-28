@echo off
echo Building the AI system...

REM Define curl path
set CURL_PATH=C:\Users\lucca\Documents\ai_tools\curl-8.12.1

REM Clean up old files
del *.o
del chatbot.exe 2>nul

REM Check if curl directories exist
echo Checking curl directories...
if not exist "%CURL_PATH%\include" (
    echo ERROR: Curl include directory not found at %CURL_PATH%\include
    goto error
)
if not exist "%CURL_PATH%\lib" (
    echo ERROR: Curl lib directory not found at %CURL_PATH%\lib
    goto error
)
if not exist "%CURL_PATH%\bin" (
    echo ERROR: Curl bin directory not found at %CURL_PATH%\bin
    goto error
)

REM List curl files for debugging
echo Listing curl lib files:
dir "%CURL_PATH%\lib" /b
echo.

REM Find the actual library file
set FOUND_CURL_LIB=0
if exist "%CURL_PATH%\lib\libcurl.a" (
    set CURL_LIB_FILE="%CURL_PATH%\lib\libcurl.a"
    set FOUND_CURL_LIB=1
    echo Found libcurl.a
)
if exist "%CURL_PATH%\lib\libcurl.dll.a" (
    set CURL_LIB_FILE="%CURL_PATH%\lib\libcurl.dll.a"
    set FOUND_CURL_LIB=1
    echo Found libcurl.dll.a
)
if exist "%CURL_PATH%\lib\libcurl.lib" (
    set CURL_LIB_FILE="%CURL_PATH%\lib\libcurl.lib"
    set FOUND_CURL_LIB=1
    echo Found libcurl.lib
)
if exist "%CURL_PATH%\lib\curl.lib" (
    set CURL_LIB_FILE="%CURL_PATH%\lib\curl.lib"
    set FOUND_CURL_LIB=1
    echo Found curl.lib
)

if %FOUND_CURL_LIB%==0 (
    echo ERROR: Could not find any curl library file!
    goto error
)

REM Compile source files with curl include path
echo Compiling source files...
g++ -std=c++11 -Wall -I"%CURL_PATH%\include" -c main.cpp
g++ -std=c++11 -Wall -I"%CURL_PATH%\include" -c neural_network.cpp
g++ -std=c++11 -Wall -I"%CURL_PATH%\include" -c memory.cpp
g++ -std=c++11 -Wall -I"%CURL_PATH%\include" -c chatbot.cpp

REM Link object files with curl library
echo Linking object files...
echo Using library file: %CURL_LIB_FILE%
g++ -o chatbot main.o neural_network.o memory.o chatbot.o %CURL_LIB_FILE% -L"%CURL_PATH%\lib"

IF %ERRORLEVEL% NEQ 0 (
    echo Direct linking failed, trying standard method...
    g++ -o chatbot main.o neural_network.o memory.o chatbot.o -L"%CURL_PATH%\lib" -lcurl
    
    IF %ERRORLEVEL% NEQ 0 (
        goto error
    )
)

REM Copy necessary DLL files
echo Copying necessary DLL files...
if exist "%CURL_PATH%\bin\libcurl.dll" (
    copy "%CURL_PATH%\bin\libcurl.dll" .
    echo Copied libcurl.dll
) else if exist "%CURL_PATH%\bin\curl.dll" (
    copy "%CURL_PATH%\bin\curl.dll" .
    echo Copied curl.dll
) else (
    echo WARNING: Could not find curl DLL file. The program may not run correctly!
    echo Listing bin directory:
    dir "%CURL_PATH%\bin" /b
)

echo Build successful!
echo Running chatbot...
chatbot
goto end

:error
echo Compilation failed!
echo.
echo Please verify that:
echo 1. libcurl is properly installed
echo 2. You have the correct MinGW/GCC version that matches the curl build
echo 3. You may need to install dependencies like OpenSSL and zlib
echo.
echo Consider using a pre-compiled binary package from:
echo https://curl.se/windows/

:end
pause