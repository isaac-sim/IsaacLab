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
rem libraries are resolvable, resolving TfType::AddAlias conflicts.
set "_ov_usd_libs_dir="
if exist "%ISAACLAB_PATH%\_isaac_sim\extscache\" (
    "%python_exe%" -c "import glob,os,sys; d=sorted(glob.glob(os.path.join(os.environ.get('ISAACLAB_PATH',''),'_isaac_sim','extscache','omni.usd.libs-*'))); print(d[-1] if d else '',end='')" > "%TEMP%\ov_usd_libs.tmp" 2>nul
    set /p _ov_usd_libs_dir=<"%TEMP%\ov_usd_libs.tmp"
    del "%TEMP%\ov_usd_libs.tmp" >nul 2>&1
)
if defined _ov_usd_libs_dir (
    set "PYTHONPATH=!_ov_usd_libs_dir!;!PYTHONPATH!"
    set "PATH=!_ov_usd_libs_dir!\bin;!PATH!"
)
set "_ov_usd_libs_dir="

rem Execute CLI.
"%python_exe%" -c "from isaaclab.cli import cli; cli()" %*

if errorlevel 1 exit /b 1
endlocal
exit /b 0
