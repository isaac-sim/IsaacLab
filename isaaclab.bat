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

rem Let Kit associate direct wrapper launches with the Isaac Sim desktop icon.
if "%RESOURCE_NAME%"=="" set "RESOURCE_NAME=IsaacSim"

rem If a local Isaac Sim binary is present, source its env setup so that
rem PYTHONPATH/PATH/EXP_PATH are correct without depending on a conda
rem activate.d hook (those don't fire under e.g. `conda run` on Windows).
if exist "%ISAACLAB_PATH%\_isaac_sim\" (
    if exist "%ISAACLAB_PATH%\_isaac_sim\setup_conda_env.bat" (
        call "%ISAACLAB_PATH%\_isaac_sim\setup_conda_env.bat" >NUL
    ) else (
        echo [WARNING] _isaac_sim is present but _isaac_sim\setup_conda_env.bat is missing; Isaac Sim env vars not exported. 1>&2
        echo [WARNING] Re-extract the Isaac Sim Windows zip if you intend to use the bundled binary. 1>&2
    )
)

rem If omni.usd.libs is present, prepend it to PYTHONPATH and its bin\ to PATH
rem so its patched pxr is found before usd-core's pxr and the native USD shared
rem libraries are resolvable.  This resolves TfType::AddAlias conflicts when
rem omni.usd.schema packages (ovrtx/ovphysx) are also installed.
rem
rem omni.usd.libs\pxr\ is a namespace package (no __init__.py), but Python
rem prefers a regular package (with __init__.py) over a namespace package
rem regardless of sys.path order.  Since usd-core installs a regular pxr
rem package, prepending omni.usd.libs to PYTHONPATH alone does nothing.
rem We write a minimal __init__.py to promote it to a regular package so the
rem PYTHONPATH prepend actually takes effect.
"%python_exe%" "%ISAACLAB_PATH%\tools\setup_usd_libs.py" > "%TEMP%\_isaaclab_usd_libs.tmp"
if errorlevel 1 (
    echo [WARNING] Failed to configure omni.usd.libs USD path; continuing without it. 1>&2
) else (
    set /p _usd_libs_dir=<"%TEMP%\_isaaclab_usd_libs.tmp"
    if defined _usd_libs_dir (
        set "PYTHONPATH=!_usd_libs_dir!;!PYTHONPATH!"
        set "PATH=!_usd_libs_dir!\bin;!PATH!"
    )
    set "_usd_libs_dir="
)
del "%TEMP%\_isaaclab_usd_libs.tmp" 2>nul


rem Execute CLI.
"%python_exe%" -c "from isaaclab.cli import cli; cli()" %*

if errorlevel 1 exit /b 1
endlocal
exit /b 0
