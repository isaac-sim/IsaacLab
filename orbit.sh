#!/bin/bash

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
extract_isaacsim_python() {
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
    local python_exe=${build_path}/python.sh
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
    python_exe=$(extract_isaacsim_python)
    # if the directory contains setup.py then install the python module
    if [ -f "$1/setup.py" ];
    then
        echo -e "\t module: $1"
        ${python_exe} -m pip install --editable $1
    fi
}

# print the usage description
print_help () {
    echo -e "\nusage: $(basename "$0") [-h] [-i] [-e] [-f] [-p] [-s] -- Utility to manage extensions in Isaac Orbit."
    echo -e "\noptional arguments:"
    echo -e "\t-h, --help       Display the help content."
    echo -e "\t-i, --install    Install the extensions inside Isaac Orbit."
    echo -e "\t-e, --extra      Install extra dependencies such as the learning frameworks."
    echo -e "\t-f, --format     Run pre-commit to format the code and check lints."
    echo -e "\t-p, --python     Run the python executable (python.sh) provided by Isaac Sim."
    echo -e "\t-s, --sim        Run the simulator executable (isaac-sim.sh) provided by Isaac Sim."
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
            export -f extract_isaacsim_python
            export -f install_orbit_extension
            # source directory
            find -L "${ORBIT_PATH}/source/extensions" -mindepth 1 -maxdepth 1 -type d -exec bash -c 'install_orbit_extension "{}"' \;
            # unset local variables
            unset install_orbit_extension
            shift # past argument
            ;;
        -e|--extra)
            # install the python packages for supported reinforcement learning frameworks
            echo "[INFO] Installing extra requirements such as learning frameworks..."
            python_exe=$(extract_isaacsim_python)
            # install the rl-frameworks specified
            ${python_exe} -m pip install -e ${ORBIT_PATH}/source/extensions/omni.isaac.orbit_envs[all]
            shift # past argument
            ;;
        -p|--python)
            # run the python provided by isaacsim
            python_exe=$(extract_isaacsim_python)
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
        -f|--format)
            # run the formatter over the repository
            # check if pre-commit is installed
            if ! command -v pre-commit &>/dev/null; then
                echo "[INFO] Installing pre-commit..."
                pip install pre-commit
            fi
            echo "[INFO] Formatting the repository..."
            # always execute inside the Orbit directory
            cd "${ORBIT_PATH}"
            pre-commit run --all-files
            cd -
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
