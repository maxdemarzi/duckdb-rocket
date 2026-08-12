@echo off
REM Build the rocket extension against the pinned DuckDB submodule.
REM
REM A .bat rather than a Makefile because vcvars64 is a batch script that mutates the
REM environment of the shell that calls it, and that environment does not survive being set up
REM in one process and used in another. The DuckDB extension template's Makefile assumes a
REM Unix-ish shell; this is the equivalent for the toolchain PLAN.md actually pins.
REM
REM MSVC Build Tools are required. clang alone cannot link a CPython/DuckDB extension on
REM Windows -- no Windows SDK -- which is a lesson already paid for once in tabicl.

setlocal

set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" (
    echo ERROR: vcvars64.bat not found at "%VCVARS%"
    exit /b 1
)
call "%VCVARS%" >nul || exit /b 1

REM cmake and ninja are installed but not on the default PATH in a non-interactive shell.
set "PATH=C:\Program Files\CMake\bin;%LOCALAPPDATA%\Microsoft\WinGet\Links;%PATH%"

REM Resolved to a canonical absolute path, not "...\scripts\..": the test runner registers
REM tests under the canonical path, and an unnormalised one matches nothing.
for %%I in ("%~dp0..") do set "ROOT=%%~fI"
set "BUILD=%ROOT%\build\release"

if "%1"=="clean" (
    if exist "%BUILD%" rmdir /s /q "%BUILD%"
    echo cleaned %BUILD%
    exit /b 0
)

REM `tests` also builds DuckDB's sqllogictest runner so test\sql\*.test can actually be
REM executed. It is off by default because the runner roughly doubles the build, and the
REM extension itself does not need it.
set "UNITTESTS=0"
if "%1"=="tests" set "UNITTESTS=1"

cmake -G Ninja -B "%BUILD%" -S "%ROOT%\duckdb" ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DDUCKDB_EXTENSION_CONFIGS="%ROOT%\extension_config.cmake" ^
    -DEXTENSION_STATIC_BUILD=1 ^
    -DBUILD_UNITTESTS=%UNITTESTS% ^
    -DBUILD_SHELL=1 || exit /b 1

cmake --build "%BUILD%" --config Release || exit /b 1

echo.
echo BUILD OK
echo   shell:     %BUILD%\duckdb.exe
echo   extension: %BUILD%\extension\rocket\rocket.duckdb_extension

if "%UNITTESTS%"=="1" (
    echo.
    echo Running test/sql/rocket.test

    REM The runner registers tests under a FORWARD-slash absolute path, so a native Windows
    REM path matches nothing. It then exits 0 having run nothing, which reads as success --
    REM so the result is checked for the pass line rather than trusted to the exit code.
    set "TESTPATH=%ROOT%\test\sql\rocket.test"
    call set "TESTPATH=%%TESTPATH:\=/%%"
    call "%BUILD%\test\unittest.exe" "%%TESTPATH%%" > "%BUILD%\rocket_test.log" 2>&1
    type "%BUILD%\rocket_test.log"
    findstr /c:"All tests passed" "%BUILD%\rocket_test.log" >nul || (
        echo TESTS DID NOT PASS ^(or matched nothing^)
        exit /b 1
    )
)
