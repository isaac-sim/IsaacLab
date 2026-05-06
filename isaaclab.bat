@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem Copyright (c) 2022-2025, The Isaac Lab Project Developers
rem (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
rem All rights reserved.
rem
rem SPDX-License-Identifier: BSD-3-Clause

set "ISAACLAB_PATH=%~dp0"
rem Remove trailing backslash.
if "%ISAACLAB_PATH:~-1%"=="\" set "ISAACLAB_PATH=%ISAACLAB_PATH:~0,-1%"

rem Find python to run CLI.
if defined VIRTUAL_ENV (
    set "python_exe=%VIRTUAL_ENV%\Scripts\python.exe"
) else if defined CONDA_PREFIX (
    set "python_exe=%CONDA_PREFIX%\python.exe"
) else if exist "%ISAACLAB_PATH%\_isaac_sim\python.bat" (
    set "python_exe=%ISAACLAB_PATH%\_isaac_sim\python.bat"
) else (
    rem Fallback.
    set "python_exe=python"
)

rem Add source/isaaclab to PYTHONPATH so we can import isaaclab.cli.
set "PYTHONPATH=%ISAACLAB_PATH%\source\isaaclab;%PYTHONPATH%"

rem If a local Isaac Sim binary is present, source its env setup so that
rem PYTHONPATH/PATH/EXP_PATH are correct without depending on a conda
rem activate.d hook (those don't fire under e.g. `conda run` on Windows).
if exist "%ISAACLAB_PATH%\_isaac_sim\setup_conda_env.bat" (
    rem Primary: direct call (works when setup_conda_env.bat does not use setlocal).
    call "%ISAACLAB_PATH%\_isaac_sim\setup_conda_env.bat" >NUL 2>&1
    rem Belt-and-suspenders: re-capture env vars via a fresh cmd subprocess.
    rem This handles cases where setup_conda_env.bat uses setlocal internally,
    rem which would cause the direct `call` above to discard its env changes.
    set "_IL_ISIM_SETUP=%ISAACLAB_PATH%\_isaac_sim\setup_conda_env.bat"
    for /f "usebackq delims=" %%E in (`cmd.exe /c "call ""%_IL_ISIM_SETUP%"" >nul 2>&1 ^&^& set"`) do set "%%E"
    set "_IL_ISIM_SETUP="
)

rem Execute CLI.
"%python_exe%" -c "from isaaclab.cli import cli; cli()" %*

if errorlevel 1 exit /b 1
endlocal
exit /b 0
