#!/usr/bin/env bash

# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

#==
# Configurations
#==

# Exits if error occurs
set -e

# Set tab-spaces
tabs 4

# get source directory
export ORBIT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

#==
# Helper functions
#==

# extract the python from isaacsim
extract_python_exe() {
    # Check if IsaacSim directory manually specified
    # Note: for manually build isaacsim, this: _build/linux-x86_64/release
    if [ ! -z ${ISAACSIM_PATH} ];
    then
        # Use local build
        build_path=${ISAACSIM_PATH}
    else
        # Use TeamCity build
        build_path=${ORBIT_PATH}/_isaac_sim
    fi
    # check if using conda
    if ! [[ -z "${CONDA_PREFIX}" ]]; then
        # use conda python
        local python_exe=${CONDA_PREFIX}/bin/python
    else
        # use python from kit
        local python_exe=${build_path}/python.sh
    fi
    # check if there is a python path available
    if [ ! -f "${python_exe}" ]; then
        echo "[ERROR] No python executable found at path: ${build_path}" >&2
        exit 1
    fi
    # return the result
    echo ${python_exe}
}

# extract the simulator exe from isaacsim
extract_isaacsim_exe() {
    # Check if IsaacSim directory manually specified
    # Note: for manually build isaacsim, this: _build/linux-x86_64/release
    if [ ! -z ${ISAACSIM_PATH} ];
    then
        # Use local build
        build_path=${ISAACSIM_PATH}
    else
        # Use TeamCity build
        build_path=${ORBIT_PATH}/_isaac_sim
    fi
    # python executable to use
    local isaacsim_exe=${build_path}/isaac-sim.sh
    # check if there is a python path available
    if [ ! -f "${isaacsim_exe}" ]; then
        echo "[ERROR] No isaac-sim executable found at path: ${build_path}" >&2
        exit 1
    fi
    # return the result
    echo ${isaacsim_exe}
}

# check if input directory is a python extension and install the module
install_orbit_extension() {
    # retrieve the python executable
    python_exe=$(extract_python_exe)
    # if the directory contains setup.py then install the python module
    if [ -f "$1/setup.py" ];
    then
        echo -e "\t module: $1"
        ${python_exe} -m pip install --editable $1
    fi
}

# setup anaconda environment for orbit
setup_conda_env() {
    # get environment name from input
    local env_name=$1
    # check conda is installed
    if ! command -v conda &> /dev/null
    then
        echo "[ERROR] Conda could not be found. Please install conda and try again."
        exit 1
    fi
    # Check if IsaacSim directory manually specified
    # Note: for manually build isaacsim, this: _build/linux-x86_64/release
    if [ ! -z ${ISAACSIM_PATH} ];
    then
        # Use local build
        build_path=${ISAACSIM_PATH}
    else
        # Use TeamCity build
        build_path=${ORBIT_PATH}/_isaac_sim
    fi
    # check if the environment exists
    if { conda env list | grep -w ${env_name}; } >/dev/null 2>&1; then
        echo -e "[INFO] Conda environment named '${env_name}' already exists."
    else
        echo -e "[INFO] Creating conda environment named '${env_name}'..."
        conda env create --name ${env_name} -f ${build_path}/environment.yml
    fi
    # cache current paths for later
    cache_pythonpath=$PYTHONPATH
    cache_ld_library_path=$LD_LIBRARY_PATH
    # clear any existing files
    rm -f ${CONDA_PREFIX}/etc/conda/activate.d/setenv.sh
    rm -f ${CONDA_PREFIX}/etc/conda/deactivate.d/unsetenv.sh
    # activate the environment
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate ${env_name}
    # setup directories to load isaac-sim variables
    mkdir -p ${CONDA_PREFIX}/etc/conda/activate.d
    mkdir -p ${CONDA_PREFIX}/etc/conda/deactivate.d
    # add variables to environment during activation
    local isaacsim_setup_conda_env_script=${ORBIT_PATH}/_isaac_sim/setup_conda_env.sh
    printf '%s\n' '#!/usr/bin/env bash' '' \
        '# for isaac-sim' \
        'source '${isaacsim_setup_conda_env_script}'' \
        '' \
        '# for orbit' \
        'export ORBIT_PATH='${ORBIT_PATH}'' \
        'alias orbit='${ORBIT_PATH}'/orbit.sh' \
        '' \
        '# show icon if not runninng headless' \
        'export RESOURCE_NAME="IsaacSim"' \
        '' > ${CONDA_PREFIX}/etc/conda/activate.d/setenv.sh
    # reactivate the environment to load the variables
    # needed because deactivate complains about orbit alias since it otherwise doesn't exist
    conda activate ${env_name}
    # remove variables from environment during deactivation
    printf '%s\n' '#!/usr/bin/env bash' '' \
        '# for orbit' \
        'unalias orbit &>/dev/null' \
        'unalias ORBIT_PATH &>/dev/null' \
        '' \
        '# for isaac-sim' \
        'unset CARB_APP_PATH' \
        'unset EXP_PATH' \
        'unset ISAAC_PATH' \
        'unset RESOURCE_NAME' \
        '' \
        '# restore paths' \
        'export PYTHONPATH='${cache_pythonpath}'' \
        'export LD_LIBRARY_PATH='${cache_ld_library_path}'' \
        '' > ${CONDA_PREFIX}/etc/conda/deactivate.d/unsetenv.sh
    # install some extra dependencies
    echo -e "[INFO] Installing extra dependencies (this might take a few minutes)..."
    conda install -c conda-forge -y importlib_metadata &> /dev/null
    # deactivate the environment
    conda deactivate
    # add information to the user about alias
    echo -e "[INFO] Added 'orbit' alias to conda environment for 'orbit.sh' script."
    echo -e "[INFO] Created conda environment named '${env_name}'.\n"
    echo -e "\t\t1. To activate the environment, run:                conda activate ${env_name}"
    echo -e "\t\t2. To install orbit extensions, run:                orbit -i"
    echo -e "\t\t3. To install learning-related dependencies, run:   orbit -e"
    echo -e "\t\t4. To perform formatting, run:                      orbit -f"
    echo -e "\t\t5. To deactivate the environment, run:              conda deactivate"
    echo -e "\n"
}

# update the vscode settings from template and isaac sim settings
update_vscode_settings() {
    echo "[INFO] Setting up vscode settings..."
    # retrieve the python executable
    python_exe=$(extract_python_exe)
    # path to setup_vscode.py
    setup_vscode_script="${ORBIT_PATH}/.vscode/tools/setup_vscode.py"
    # check if the file exists before attempting to run it
    if [ -f "${setup_vscode_script}" ]; then
        ${python_exe} "${setup_vscode_script}"
    else
        echo "[WARNING] setup_vscode.py not found. Aborting vscode settings setup."
    fi
}

# print the usage description
print_help () {
    echo -e "\nusage: $(basename "$0") [-h] [-i] [-e] [-f] [-p] [-s] [-t] [-o] [-v] [-d] [-c] -- Utility to manage Orbit."
    echo -e "\noptional arguments:"
    echo -e "\t-h, --help           Display the help content."
    echo -e "\t-i, --install        Install the extensions inside Orbit."
    echo -e "\t-e, --extra [LIB]    Install learning frameworks (rl_games, rsl_rl, sb3) as extra dependencies. Default is 'all'."
    echo -e "\t-f, --format         Run pre-commit to format the code and check lints."
    echo -e "\t-p, --python         Run the python executable provided by Isaac Sim or virtual environment (if active)."
    echo -e "\t-s, --sim            Run the simulator executable (isaac-sim.sh) provided by Isaac Sim."
    echo -e "\t-t, --test           Run all python unittest tests."
    echo -e "\t-o, --docker         Run the docker container helper script (docker/container.sh)."
    echo -e "\t-v, --vscode         Generate the VSCode settings file from template."
    echo -e "\t-d, --docs           Build the documentation from source using sphinx."
    echo -e "\t-c, --conda [NAME]   Create the conda environment for Orbit. Default name is 'orbit'."
    echo -e "\n" >&2
}


#==
# Main
#==

# check argument provided
if [ -z "$*" ]; then
    echo "[Error] No arguments provided." >&2;
    print_help
    exit 1
fi

# pass the arguments
while [[ $# -gt 0 ]]; do
    # read the key
    case "$1" in
        -i|--install)
            # install the python packages in omni_isaac_orbit/source directory
            echo "[INFO] Installing extensions inside orbit repository..."
            # recursively look into directories and install them
            # this does not check dependencies between extensions
            export -f extract_python_exe
            export -f install_orbit_extension
            # source directory
            find -L "${ORBIT_PATH}/source/extensions" -mindepth 1 -maxdepth 1 -type d -exec bash -c 'install_orbit_extension "{}"' \;
            # unset local variables
            unset install_orbit_extension
            # setup vscode settings
            update_vscode_settings
            shift # past argument
            ;;
        -e|--extra)
            # install the python packages for supported reinforcement learning frameworks
            echo "[INFO] Installing extra requirements such as learning frameworks..."
            python_exe=$(extract_python_exe)
            # check if specified which rl-framework to install
            if [ -z "$2" ]; then
                echo "[INFO] Installing all rl-frameworks..."
                framework_name="all"
            else
                echo "[INFO] Installing rl-framework: $2"
                framework_name=$2
                shift # past argument
            fi
            # install the rl-frameworks specified
            ${python_exe} -m pip install -e ${ORBIT_PATH}/source/extensions/omni.isaac.orbit_tasks["${framework_name}"]
            shift # past argument
            ;;
        -c|--conda)
            # use default name if not provided
            if [ -z "$2" ]; then
                echo "[INFO] Using default conda environment name: orbit"
                conda_env_name="orbit"
            else
                echo "[INFO] Using conda environment name: $2"
                conda_env_name=$2
                shift # past argument
            fi
            # setup the conda environment for orbit
            setup_conda_env ${conda_env_name}
            shift # past argument
            ;;
        -f|--format)
            # reset the python path to avoid conflicts with pre-commit
            # this is needed because the pre-commit hooks are installed in a separate virtual environment
            # and it uses the system python to run the hooks
            if [ -n "${CONDA_DEFAULT_ENV}" ]; then
                cache_pythonpath=${PYTHONPATH}
                export PYTHONPATH=""
            fi
            # run the formatter over the repository
            # check if pre-commit is installed
            if ! command -v pre-commit &>/dev/null; then
                echo "[INFO] Installing pre-commit..."
                pip install pre-commit
            fi
            # always execute inside the Orbit directory
            echo "[INFO] Formatting the repository..."
            cd ${ORBIT_PATH}
            pre-commit run --all-files
            cd - > /dev/null
            # set the python path back to the original value
            if [ -n "${CONDA_DEFAULT_ENV}" ]; then
                export PYTHONPATH=${cache_pythonpath}
            fi
            shift # past argument
            # exit neatly
            break
            ;;
        -p|--python)
            # run the python provided by isaacsim
            python_exe=$(extract_python_exe)
            echo "[INFO] Using python from: ${python_exe}"
            shift # past argument
            ${python_exe} $@
            # exit neatly
            break
            ;;
        -s|--sim)
            # run the simulator exe provided by isaacsim
            isaacsim_exe=$(extract_isaacsim_exe)
            echo "[INFO] Running isaac-sim from: ${isaacsim_exe}"
            shift # past argument
            ${isaacsim_exe} --ext-folder ${ORBIT_PATH}/source/extensions $@
            # exit neatly
            break
            ;;
        -t|--test)
            # run the python provided by isaacsim
            python_exe=$(extract_python_exe)
            shift # past argument
            ${python_exe} ${ORBIT_PATH}/tools/run_all_tests.py $@
            # exit neatly
            break
            ;;
        -o|--docker)
            # run the docker container helper script
            docker_script=${ORBIT_PATH}/docker/container.sh
            echo "[INFO] Running docker utility script from: ${docker_script}"
            shift # past argument
            bash ${docker_script} $@
            # exit neatly
            break
            ;;
        -v|--vscode)
            # update the vscode settings
            update_vscode_settings
            shift # past argument
            # exit neatly
            break
            ;;
        -d|--docs)
            # build the documentation
            echo "[INFO] Building documentation..."
            # retrieve the python executable
            python_exe=$(extract_python_exe)
            # install pip packages
            cd ${ORBIT_PATH}/docs
            ${python_exe} -m pip install -r requirements.txt > /dev/null
            # build the documentation
            ${python_exe} -m sphinx -b html -d _build/doctrees . _build/html
            # open the documentation
            echo -e "[INFO] To open documentation on default browser, run:"
            echo -e "\n\t\txdg-open $(pwd)/_build/html/index.html\n"
            # exit neatly
            cd - > /dev/null
            shift # past argument
            # exit neatly
            break
            ;;
        -h|--help)
            print_help
            exit 1
            ;;
        *) # unknown option
            echo "[Error] Invalid argument provided: $1"
            print_help
            exit 1
            ;;
    esac
done
