@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem Copyright (c) 2022-2025, The Isaac Lab Project Developers.
rem All rights reserved.
rem
rem SPDX-License-Identifier: BSD-3-Clause

rem Configurations
set "ISAACLAB_PATH=%~dp0"
goto main

rem Helper functions

rem extract Isaac Sim directory
:extract_isaacsim_path
rem Use the sym-link path to Isaac Sim directory
set isaac_path=%ISAACLAB_PATH%\_isaac_sim
rem Check if directory exists
if not exist "%isaac_path%" (
    rem Find the Python executable
    call :extract_python_exe
    rem retrieve the isaacsim path from the installed package
    set "isaac_path="
    for /f "delims=" %%i in ('!python_exe! -c "import isaacsim; import os; print(os.environ['ISAAC_PATH'])"') do (
        if not defined isaac_path (
            set "isaac_path=%%i"
        )
    )
)
rem Check if the directory exists
if not exist "%isaac_path%" (
    echo [ERROR] Unable to find the Isaac Sim directory: %isaac_path%
    echo %tab%This could be due to the following reasons:
    echo %tab%1. Conda environment with Isaac Sim pip package is not activated.
    echo %tab%2. Isaac Sim directory is not available at the default path: %ISAACLAB_PATH%\_isaac_sim
    exit /b 1
)
goto :eof

rem -----------------------------------------------------------------------
rem Returns success (exit code 0) if Isaac Sim's version starts with "4.5"
rem -----------------------------------------------------------------------
:is_isaacsim_version_4_5
    rem make sure we have %python_exe%
    call :extract_python_exe

    rem 1) try to locate the VERSION file via the kit install
    for /f "delims=" %%V in ('!python_exe! -c "import isaacsim,os;print(os.path.abspath(os.path.join(os.path.dirname(isaacsim.__file__), os.pardir, os.pardir, 'VERSION')))"') do set "VERSION_PATH=%%V"
    if exist "!VERSION_PATH!" (
        for /f "usebackq delims=" %%L in ("!VERSION_PATH!") do set "ISAACSIM_VER=%%L"
    ) else (
        rem 2) fallback to importlib.metadata if no VERSION file
        for /f "delims=" %%L in ('!python_exe! -c "from importlib.metadata import version;print(version(''isaacsim''))"') do set "ISAACSIM_VER=%%L"
    )

    rem Clean up the version string (remove any trailing whitespace or newlines)
    set "ISAACSIM_VER=!ISAACSIM_VER: =!"

    rem Use string comparison instead of findstr for more reliable matching
    if "!ISAACSIM_VER:~0,3!"=="4.5" (
        exit /b 0
    ) else (
        exit /b 1
    )
    goto :eof

rem extract the python from isaacsim
:extract_python_exe
rem check if using conda
if not "%CONDA_PREFIX%"=="" (
    rem use conda python
    set python_exe=%CONDA_PREFIX%\python.exe
) else (
    rem use kit python
    set python_exe=%ISAACLAB_PATH%\_isaac_sim\python.bat
)
rem check for if isaac sim was installed to system python
if not exist "%python_exe%" (
    set "python_exe="
    python -m pip show isaacsim-rl > nul 2>&1
    if %ERRORLEVEL% equ 0 (
        for /f "delims=" %%i in ('where python') do (
            if not defined python_exe (
                set "python_exe=%%i"
            )
        )
    )
)
if not exist "%python_exe%" (
    echo [ERROR] Unable to find any Python executable at path: %python_exe%
    echo %tab%This could be due to the following reasons:
    echo %tab%1. Conda environment is not activated.
    echo %tab%2. Python executable is not available at the default path: %ISAACLAB_PATH%\_isaac_sim\python.bat
    exit /b 1
)
goto :eof


rem extract the simulator exe from isaacsim
:extract_isaacsim_exe
call :extract_python_exe
call !python_exe! -m pip show isaacsim-rl > nul 2>&1
if errorlevel 1 (
    rem obtain isaacsim path
    call :extract_isaacsim_path
    rem python executable to use
    set isaacsim_exe=!isaac_path!\isaac-sim.bat
) else (
    rem if isaac sim installed from pip
    set isaacsim_exe=isaacsim isaacsim.exp.full
)
rem check if there is a python path available
if not exist "%isaacsim_exe%" (
    echo [ERROR] No isaac-sim executable found at path: %isaacsim_exe%
    exit /b 1
)
goto :eof


rem check if input directory is a python extension and install the module
:install_isaaclab_extension
echo %ext_folder%
rem retrieve the python executable
call :extract_python_exe
rem if the directory contains setup.py then install the python module
if exist "%ext_folder%\setup.py" (
    echo     module: %ext_folder%
    call !python_exe! -m pip install --editable %ext_folder%
)
goto :eof


rem setup anaconda environment for Isaac Lab
:setup_conda_env
rem get environment name from input
set env_name=%conda_env_name%
rem check if conda is installed
where conda >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Conda could not be found. Please install conda and try again.
    exit /b 1
)

rem check if _isaac_sim symlink exists and isaacsim-rl is not installed via pip
if not exist "%ISAACLAB_PATH%\_isaac_sim" (
    python -m pip list | findstr /C:"isaacsim-rl" >nul
    if errorlevel 1 (
        echo [WARNING] _isaac_sim symlink not found at %ISAACLAB_PATH%\_isaac_sim
        echo     This warning can be ignored if you plan to install Isaac Sim via pip.
        echo     If you are using a binary installation of Isaac Sim, please ensure the symlink is created before setting up the conda environment.
    )
)

rem check if the environment exists
call conda env list | findstr /c:"%env_name%" >nul
if %errorlevel% equ 0 (
    echo [INFO] Conda environment named '%env_name%' already exists.
) else (
    echo [INFO] Creating conda environment named '%env_name%'...
    echo [INFO] Installing dependencies from %ISAACLAB_PATH%\environment.yml
    rem ————————————————————————————————
    rem patch Python version if needed, but back up first
    rem ————————————————————————————————
    copy "%ISAACLAB_PATH%environment.yml" "%ISAACLAB_PATH%environment.yml.bak" >nul
    call :is_isaacsim_version_4_5
    if !ERRORLEVEL! EQU 0 (
        echo [INFO] Detected Isaac Sim 4.5 --^> forcing python=3.10
        rem Use findstr to replace the python version line
        (
            for /f "delims=" %%L in ('type "%ISAACLAB_PATH%environment.yml"') do (
                set "line=%%L"
                set "line=!line: =!"
                if "!line:~0,15!"=="-python=3.11" (
                    echo   - python=3.10
                ) else (
                    echo %%L
                )
            )
        ) > "%ISAACLAB_PATH%environment.yml.tmp"
        rem Replace the original file with the modified version
        move /y "%ISAACLAB_PATH%environment.yml.tmp" "%ISAACLAB_PATH%environment.yml" >nul
    ) else (
        echo [INFO] Isaac Sim ^>=5.0, installing python=3.11
    )
    call conda env create -y --file %ISAACLAB_PATH%\environment.yml -n %env_name%
)
rem cache current paths for later
set "cache_pythonpath=%PYTHONPATH%"
set "cache_ld_library_path=%LD_LIBRARY_PATH%"
rem clear any existing files
echo %CONDA_PREFIX%
del "%CONDA_PREFIX%\etc\conda\activate.d\setenv.bat" 2>nul
del "%CONDA_PREFIX%\etc\conda\deactivate.d\unsetenv.bat" 2>nul
rem activate the environment
call conda activate %env_name%
rem setup directories to load isaac-sim variables
mkdir "%CONDA_PREFIX%\etc\conda\activate.d" 2>nul
mkdir "%CONDA_PREFIX%\etc\conda\deactivate.d" 2>nul

rem obtain isaacsim path
call :extract_isaacsim_path
if exist "%isaac_path%" (
    rem add variables to environment during activation
    (
        echo @echo off
        echo rem for isaac-sim
        echo set "RESOURCE_NAME=IsaacSim"
        echo set CARB_APP_PATH=!isaac_path!\kit
        echo set EXP_PATH=!isaac_path!\apps
        echo set ISAAC_PATH=!isaac_path!
        echo set PYTHONPATH=%PYTHONPATH%;!isaac_path!\site
        echo.
        echo rem for isaac-lab
        echo doskey isaaclab=isaaclab.bat $*
    ) > "%CONDA_PREFIX%\etc\conda\activate.d\env_vars.bat"
    (
        echo $env:CARB_APP_PATH="!isaac_path!\kit"
        echo $env:EXP_PATH="!isaac_path!\apps"
        echo $env:ISAAC_PATH="!isaac_path!"
        echo $env:PYTHONPATH="%PYTHONPATH%;!isaac_path!\site"
        echo $env:RESOURCE_NAME="IsaacSim"
    ) > "%CONDA_PREFIX%\etc\conda\activate.d\env_vars.ps1"
) else (
    rem assume isaac sim will be installed from pip
    rem add variables to environment during activation
    (
        echo @echo off
        echo rem for isaac-sim
        echo set "RESOURCE_NAME=IsaacSim"
        echo.
        echo rem for isaac-lab
        echo doskey isaaclab=isaaclab.bat $*
    ) > "%CONDA_PREFIX%\etc\conda\activate.d\env_vars.bat"
    (
        echo $env:RESOURCE_NAME="IsaacSim"
    ) > "%CONDA_PREFIX%\etc\conda\activate.d\env_vars.ps1"
)

rem reactivate the environment to load the variables
call conda activate %env_name%

rem remove variables from environment during deactivation
(
    echo @echo off
    echo rem for isaac-sim
    echo set "CARB_APP_PATH="
    echo set "EXP_PATH="
    echo set "ISAAC_PATH="
    echo set "RESOURCE_NAME="
    echo.
    echo rem for isaac-lab
    echo doskey isaaclab =
    echo.
    echo rem restore paths
    echo set "PYTHONPATH=%cache_pythonpath%"
    echo set "LD_LIBRARY_PATH=%cache_ld_library_path%"
) > "%CONDA_PREFIX%\etc\conda\deactivate.d\unsetenv_vars.bat"
(
    echo $env:RESOURCE_NAME=""
    echo $env:PYTHONPATH="%cache_pythonpath%"
    echo $env:LD_LIBRARY_PATH="%cache_pythonpath%"
) > "%CONDA_PREFIX%\etc\conda\deactivate.d\unsetenv_vars.ps1"

rem deactivate the environment
call conda deactivate
rem add information to the user about alias
echo [INFO] Added 'isaaclab' alias to conda environment for 'isaaclab.bat' script.
echo [INFO] Created conda environment named '%env_name%'.
echo.
echo       1. To activate the environment, run:                conda activate %env_name%
echo       2. To install Isaac Lab extensions, run:            isaaclab -i
echo       3. To perform formatting, run:                      isaaclab -f
echo       4. To deactivate the environment, run:              conda deactivate
echo.
goto :eof


rem Update the vscode settings from template and Isaac Sim settings
:update_vscode_settings
echo [INFO] Setting up vscode settings...
rem Retrieve the python executable
call :extract_python_exe
rem Path to setup_vscode.py
set "setup_vscode_script=%ISAACLAB_PATH%\.vscode\tools\setup_vscode.py"
rem Check if the file exists before attempting to run it
if exist "%setup_vscode_script%" (
    call !python_exe! "%setup_vscode_script%"
) else (
    echo [WARNING] setup_vscode.py not found. Aborting vscode settings setup.
)
goto :eof


rem Print the usage description
:print_help
echo.
echo usage: %~nx0 [-h] [-i] [-f] [-p] [-s] [-v] [-d] [-n] [-c] -- Utility to manage extensions in Isaac Lab.
echo.
echo optional arguments:
echo     -h, --help           Display the help content.
echo     -i, --install [LIB]  Install the extensions inside Isaac Lab and learning frameworks as extra dependencies. Default is 'all'.
echo     -f, --format         Run pre-commit to format the code and check lints.
echo     -p, --python         Run the python executable (python.bat) provided by Isaac Sim.
echo     -s, --sim            Run the simulator executable (isaac-sim.bat) provided by Isaac Sim.
echo     -t, --test           Run all python pytest tests.
echo     -v, --vscode         Generate the VSCode settings file from template.
echo     -d, --docs           Build the documentation from source using sphinx.
echo     -n, --new            Create a new external project or internal task from template.
echo     -c, --conda [NAME]   Create the conda environment for Isaac Lab. Default name is 'env_isaaclab'.
echo.
goto :eof


rem Main
:main

rem check argument provided
if "%~1"=="" (
    echo [Error] No arguments provided.
    call :print_help
    exit /b 1
)

rem pass the arguments
:loop
if "%~1"=="" goto :end
set "arg=%~1"

rem read the key
if "%arg%"=="-i" (
    rem install the python packages in isaaclab/source directory
    echo [INFO] Installing extensions inside the Isaac Lab repository...
    call :extract_python_exe
    rem check if pytorch is installed and its version
    rem install pytorch with cuda 12.8 for blackwell support
    call !python_exe! -m pip list | findstr /C:"torch" >nul
    if %errorlevel% equ 0 (
        for /f "tokens=2" %%i in ('!python_exe! -m pip show torch ^| findstr /C:"Version:"') do (
            set torch_version=%%i
        )
        if not "!torch_version!"=="2.7.0+cu128" (
            echo [INFO] Uninstalling PyTorch version !torch_version!...
            call !python_exe! -m pip uninstall -y torch torchvision torchaudio
            echo [INFO] Installing PyTorch 2.7.0 with CUDA 12.8 support...
            call !python_exe! -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
        ) else (
            echo [INFO] PyTorch 2.7.0 is already installed.
        )
    ) else (
        echo [INFO] Installing PyTorch 2.7.0 with CUDA 12.8 support...
        call !python_exe! -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
    )

    for /d %%d in ("%ISAACLAB_PATH%\source\*") do (
        set ext_folder="%%d"
        call :install_isaaclab_extension
    )
    rem install the python packages for supported reinforcement learning frameworks
    echo [INFO] Installing extra requirements such as learning frameworks...
    if "%~2"=="" (
        echo [INFO] Installing all rl-frameworks.
        set framework_name=all
    ) else if "%~2"=="none" (
        echo [INFO]  No rl-framework will be installed.
        set framework_name=none
        shift
    ) else (
        echo [INFO] Installing rl-framework: %2.
        set framework_name=%2
        shift
    )
    rem install the rl-frameworks specified
    call !python_exe! -m pip install -e %ISAACLAB_PATH%\source\isaaclab_rl[!framework_name!]
    shift
) else if "%arg%"=="--install" (
    rem install the python packages in source directory
    echo [INFO] Installing extensions inside the Isaac Lab repository...
    call :extract_python_exe

    rem check if pytorch is installed and its version
    rem install pytorch with cuda 12.8 for blackwell support
    call !python_exe! -m pip list | findstr /C:"torch" >nul
    if %errorlevel% equ 0 (
        for /f "tokens=2" %%i in ('!python_exe! -m pip show torch ^| findstr /C:"Version:"') do (
            set torch_version=%%i
        )
        if not "!torch_version!"=="2.7.0+cu128" (
            echo [INFO] Uninstalling PyTorch version !torch_version!...
            call !python_exe! -m pip uninstall -y torch torchvision torchaudio
            echo [INFO] Installing PyTorch 2.7.0 with CUDA 12.8 support...
            call !python_exe! -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
        ) else (
            echo [INFO] PyTorch 2.7.0 is already installed.
        )
    ) else (
        echo [INFO] Installing PyTorch 2.7.0 with CUDA 12.8 support...
        call !python_exe! -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
    )

    for /d %%d in ("%ISAACLAB_PATH%\source\*") do (
        set ext_folder="%%d"
        call :install_isaaclab_extension
    )
    rem install the python packages for supported reinforcement learning frameworks
    echo [INFO] Installing extra requirements such as learning frameworks...
    if "%~2"=="" (
        echo [INFO] Installing all rl-frameworks.
        set framework_name=all
    ) else if "%~2"=="none" (
        echo [INFO]  No rl-framework will be installed.
        set framework_name=none
        shift
    ) else (
        echo [INFO] Installing rl-framework: %2.
        set framework_name=%2
        shift
    )
    rem install the rl-frameworks specified
    call !python_exe! -m pip install -e %ISAACLAB_PATH%\source\isaaclab_rl[!framework_name!]
    rem update the vscode settings
    rem once we have a docker container, we need to disable vscode settings
    call :update_vscode_settings
    shift
) else if "%arg%"=="-c" (
    rem use default name if not provided
    if not "%~2"=="" (
        echo [INFO] Using conda environment name: %2
        set conda_env_name=%2
        shift
    ) else (
        echo [INFO] Using default conda environment name: env_isaaclab
        set conda_env_name=env_isaaclab
    )
    call :setup_conda_env %conda_env_name%
    shift
) else if "%arg%"=="--conda" (
    rem use default name if not provided
    if not "%~2"=="" (
        echo [INFO] Using conda environment name: %2
        set conda_env_name=%2
        shift
    ) else (
        echo [INFO] Using default conda environment name: env_isaaclab
        set conda_env_name=env_isaaclab
    )
    call :setup_conda_env %conda_env_name%
    shift
) else if "%arg%"=="-f" (
    rem reset the python path to avoid conflicts with pre-commit
    rem this is needed because the pre-commit hooks are installed in a separate virtual environment
    rem and it uses the system python to run the hooks
    if not "%CONDA_DEFAULT_ENV%"=="" (
        set cache_pythonpath=%PYTHONPATH%
        set PYTHONPATH=
    )

    rem run the formatter over the repository
    rem check if pre-commit is installed
    pip show pre-commit > nul 2>&1
    if errorlevel 1 (
        echo [INFO] Installing pre-commit...
        pip install pre-commit
    )

    rem always execute inside the Isaac Lab directory
    echo [INFO] Formatting the repository...
    pushd %ISAACLAB_PATH%
    call python -m pre_commit run --all-files
    popd >nul

    rem set the python path back to the original value
    if not "%CONDA_DEFAULT_ENV%"=="" (
        set PYTHONPATH=%cache_pythonpath%
    )
    goto :end
) else if "%arg%"=="--format" (
    rem reset the python path to avoid conflicts with pre-commit
    rem this is needed because the pre-commit hooks are installed in a separate virtual environment
    rem and it uses the system python to run the hooks
    if not "%CONDA_DEFAULT_ENV%"=="" (
        set cache_pythonpath=%PYTHONPATH%
        set PYTHONPATH=
    )

    rem run the formatter over the repository
    rem check if pre-commit is installed
    pip show pre-commit > nul 2>&1
    if errorlevel 1 (
        echo [INFO] Installing pre-commit...
        pip install pre-commit
    )

    rem always execute inside the Isaac Lab directory
    echo [INFO] Formatting the repository...
    pushd %ISAACLAB_PATH%
    call python -m pre_commit run --all-files
    popd >nul

    rem set the python path back to the original value
    if not "%CONDA_DEFAULT_ENV%"=="" (
        set PYTHONPATH=%cache_pythonpath%
    )
    goto :end
) else if "%arg%"=="-p" (
    rem run the python provided by Isaac Sim
    call :extract_python_exe
    echo [INFO] Using python from: !python_exe!
    REM Loop through all arguments - mimic shift
    set "allArgs="
    for %%a in (%*) do (
        REM Append each argument to the variable, skip the first one
        if defined skip (
            set "allArgs=!allArgs! %%a"
        ) else (
            set "skip=1"
        )
    )
    call !python_exe! !allArgs!
    goto :end
) else if "%arg%"=="--python" (
    rem run the python provided by Isaac Sim
    call :extract_python_exe
    echo [INFO] Using python from: !python_exe!
    REM Loop through all arguments - mimic shift
    set "allArgs="
    for %%a in (%*) do (
        REM Append each argument to the variable, skip the first one
        if defined skip (
            set "allArgs=!allArgs! %%a"
        ) else (
            set "skip=1"
        )
    )
    call !python_exe! !allArgs!
    goto :end
) else if "%arg%"=="-s" (
    rem run the simulator exe provided by isaacsim
    call :extract_isaacsim_exe
    echo [INFO] Running isaac-sim from: !isaacsim_exe!
    set "allArgs="
    for %%a in (%*) do (
        REM Append each argument to the variable, skip the first one
        if defined skip (
            set "allArgs=!allArgs! %%a"
        ) else (
            set "skip=1"
        )
    )
    !isaacsim_exe! --ext-folder %ISAACLAB_PATH%\source !allArgs1
    goto :end
) else if "%arg%"=="--sim" (
    rem run the simulator exe provided by Isaac Sim
    call :extract_isaacsim_exe
    echo [INFO] Running isaac-sim from: !isaacsim_exe!
    set "allArgs="
    for %%a in (%*) do (
        REM Append each argument to the variable, skip the first one
        if defined skip (
            set "allArgs=!allArgs! %%a"
        ) else (
            set "skip=1"
        )
    )
    !isaacsim_exe! --ext-folder %ISAACLAB_PATH%\source !allArgs1
    goto :end
) else if "%arg%"=="-n" (
    rem run the template generator script
    call :extract_python_exe
    set "allArgs="
    for %%a in (%*) do (
        REM Append each argument to the variable, skip the first one
        if defined skip (
            set "allArgs=!allArgs! %%a"
        ) else (
            set "skip=1"
        )
    )
    echo [INFO] Installing template dependencies...
    call !python_exe! -m pip install -q -r tools\template\requirements.txt
    echo.
    echo [INFO] Running template generator...
    echo.
    call !python_exe! tools\template\cli.py !allArgs!
    goto :end
) else if "%arg%"=="--new" (
    rem run the template generator script
    call :extract_python_exe
    set "allArgs="
    for %%a in (%*) do (
        REM Append each argument to the variable, skip the first one
        if defined skip (
            set "allArgs=!allArgs! %%a"
        ) else (
            set "skip=1"
        )
    )
    echo [INFO] Installing template dependencies...
    call !python_exe! -m pip install -q -r tools\template\requirements.txt
    echo.
    echo [INFO] Running template generator...
    echo.
    call !python_exe! tools\template\cli.py !allArgs!
    goto :end
) else if "%arg%"=="-t" (
    rem run the python provided by Isaac Sim
    call :extract_python_exe
    set "allArgs="
    for %%a in (%*) do (
        REM Append each argument to the variable, skip the first one
        if defined skip (
            set "allArgs=!allArgs! %%a"
        ) else (
            set "skip=1"
        )
    )
    call !python_exe! -m pytest tools !allArgs!
    goto :end
) else if "%arg%"=="--test" (
    rem run the python provided by Isaac Sim
    call :extract_python_exe
    set "allArgs="
    for %%a in (%*) do (
        REM Append each argument to the variable, skip the first one
        if defined skip (
            set "allArgs=!allArgs! %%a"
        ) else (
            set "skip=1"
        )
    )
    call !python_exe! -m pytest tools !allArgs!
    goto :end
) else if "%arg%"=="-v" (
    rem update the vscode settings
    call :update_vscode_settings
    shift
    goto :end
) else if "%arg%"=="--vscode" (
    rem update the vscode settings
    call :update_vscode_settings
    shift
    goto :end
) else if "%arg%"=="-d" (
    rem build the documentation
    echo [INFO] Building documentation...
    call :extract_python_exe
    pushd %ISAACLAB_PATH%\docs
    call !python_exe! -m pip install -r requirements.txt >nul
    call !python_exe! -m sphinx -b html -d _build\doctrees . _build\html
    echo [INFO] To open documentation on default browser, run:
    echo xdg-open "%ISAACLAB_PATH%\docs\_build\html\index.html"
    popd >nul
    shift
    goto :end
) else if "%arg%"=="--docs" (
    rem build the documentation
    echo [INFO] Building documentation...
    call :extract_python_exe
    pushd %ISAACLAB_PATH%\docs
    call !python_exe! -m pip install -r requirements.txt >nul
    call !python_exe! -m sphinx -b html -d _build\doctrees . _build\current
    echo [INFO] To open documentation on default browser, run:
    echo xdg-open "%ISAACLAB_PATH%\docs\_build\current\index.html"
    popd >nul
    shift
    goto :end
) else if "%arg%"=="-h" (
    call :print_help
    goto :end
) else if "%arg%"=="--help" (
    call :print_help
    goto :end
) else (
    echo Invalid argument provided: %arg%
    call :print_help
    exit /b 1
)
goto loop

:end
exit /b 0
