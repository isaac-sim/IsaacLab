#!/usr/bin/env bash

#==
# Configurations
#==

# Exits if error occurs
set -e

# Set tab-spaces
tabs 4

# get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

#==
# Functions
#==

# print the usage description
print_help () {
    echo -e "\nusage: $(basename "$0") [-h] [run] [start] [stop] -- Utility for handling docker in Orbit."
    echo -e "\noptional arguments:"
    echo -e "\t-h, --help                  Display the help content."
    echo -e "\tstart [profile]             Build the docker image and create the container in detached mode."
    echo -e "\tenter [profile]             Begin a new bash process within an existing orbit container."
    echo -e "\tcopy [profile]              Copy build and logs artifacts from the container to the host machine."
    echo -e "\tstop [profile]              Stop the docker container and remove it."
    echo -e "\tpush [profile]              Push the docker image to the cluster."
    echo -e "\tjob [profile] [job_args]    Submit a job to the cluster."
    echo -e "\n"
    echo -e "[profile] is the optional container profile specification and [job_args] optional arguments specific"
    echo -e "to the executed script"
    echo -e "\n" >&2
}

install_apptainer() {
    # Installation procedure from here: https://apptainer.org/docs/admin/main/installation.html#install-ubuntu-packages
    read -p "[INFO] Required 'apptainer' package could not be found. Would you like to install it via apt? (y/N)" app_answer
    if [ "$app_answer" != "${app_answer#[Yy]}" ]; then
        sudo apt update && sudo apt install -y software-properties-common
        sudo add-apt-repository -y ppa:apptainer/ppa
        sudo apt update && sudo apt install -y apptainer
    else
        echo "[INFO] Exiting because apptainer was not installed"
        exit
    fi
}

# Function to check docker versions
# If docker version is more than 25, the script errors out.
check_docker_version() {
    # Retrieve Docker version
    docker_version=$(docker --version | awk '{ print $3 }')
    apptainer_version=$(apptainer --version | awk '{ print $3 }')

    # Check if version is above 25.xx
    if [ "$(echo "${docker_version}" | cut -d '.' -f 1)" -ge 25 ]; then
        echo "[ERROR]: Docker version ${docker_version} is not compatible with Apptainer version ${apptainer_version}. Exiting."
        exit 1
    else
        echo "[INFO]: Building singularity with docker version: ${docker_version} and Apptainer version: ${apptainer_version}."
    fi
}

# Produces container_profile, add_profiles, and add_envs from the image_extension arg
resolve_image_extension() {
    # If no profile was passed, we default to 'base'
    container_profile=${1:-"base"}
    # check if the second argument has to be a profile or can be a job argument instead
    necessary_profile=${2:-true}

    # We also default to 'base' if "orbit" is passed
    if [ "$1" == "orbit" ]; then
        container_profile="base"
    fi

    # check if a .env.$container_profile file exists
    # if the argument is necessary a profile, then the file must exists otherwise an info is printed
    if [ "$necessary_profile" = true ] && [ ! -f $SCRIPT_DIR/.env.$container_profile ]; then
        echo "[Error] The profile '$container_profile' has no .env.$container_profile file!" >&2;
        exit 1
    elif [ ! -f $SCRIPT_DIR/.env.$container_profile ]; then
        echo "[INFO] No .env.$container_profile found, assume second argument is no profile! Will use default container!" >&2;
        container_profile="base"
    fi

    add_profiles="--profile $container_profile"
    # We will need .env.base regardless of profile
    add_envs="--env-file .env.base"
    # The second argument is interpreted as the profile to use.
    # We will select the base profile by default.
    # This will also determine the .env file that is loaded
    if [ "$container_profile" != "base" ]; then
        # We have to load multiple .env files here in order to combine
        # them for the args from base required for extensions, (i.e. DOCKER_USER_HOME)
        add_envs="$add_envs --env-file .env.$container_profile"
    fi
}

# Prints a warning message and exits if the passed container is not running
is_container_running() {
    container_name="$1"
    if [ "$( docker container inspect -f '{{.State.Status}}' $container_name 2> /dev/null)" != "running" ]; then
        echo "[Error] The '$container_name' container is not running!" >&2;
        exit 1
    fi
}

# Checks if a docker image exists, otherwise prints warning and exists
check_image_exists() {
    image_name="$1"
    if ! docker image inspect $image_name &> /dev/null; then
        echo "[Error] The '$image_name' image does not exist!" >&2;
        exit 1
    fi
}

# Check if the singularity image exists on the remote host, otherwise print warning and exit
check_singularity_image_exists() {
    image_name="$1"
    if ! ssh "$CLUSTER_LOGIN" "[ -f $CLUSTER_SIF_PATH/$image_name.tar ]"; then
        echo "[Error] The '$image_name' image does not exist on the remote host $CLUSTER_LOGIN!" >&2;
        exit 1
    fi
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

# check if docker is installed
if ! command -v docker &> /dev/null; then
    echo "[Error] Docker is not installed! Please check the 'Docker Guide' for instruction." >&2;
    exit 1
fi

# parse arguments
mode="$1"
profile_arg="$2" # Capture the second argument as the potential profile argument

# Check mode argument and resolve the container profile
case $mode in
    build|start|enter|copy|stop|push)
        resolve_image_extension "$profile_arg" true
        ;;
    job)
        resolve_image_extension "$profile_arg" false
        ;;
    *)
        # Not recognized mode
        echo "[Error] Invalid command provided: $mode"
        print_help
        exit 1
        ;;
esac

# Produces a nice print statement stating which container profile is being used
echo "[INFO] Using container profile: $container_profile"

# resolve mode
case $mode in
    start)
        echo "[INFO] Building the docker image and starting the container orbit-$container_profile in the background..."
        pushd ${SCRIPT_DIR} > /dev/null 2>&1
        # We have to build the base image as a separate step,
        # in case we are building a profile which depends
        # upon
        docker compose --file docker-compose.yaml --env-file .env.base build orbit-base
        docker compose --file docker-compose.yaml $add_profiles $add_envs up --detach --build --remove-orphans
        popd > /dev/null 2>&1
        ;;
    enter)
        # Check that desired container is running, exit if it isn't
        is_container_running orbit-$container_profile
        echo "[INFO] Entering the existing 'orbit-$container_profile' container in a bash session..."
        pushd ${SCRIPT_DIR} > /dev/null 2>&1
        docker exec --interactive --tty orbit-$container_profile bash
        popd > /dev/null 2>&1
        ;;
    copy)
        # Check that desired container is running, exit if it isn't
        is_container_running orbit-$container_profile
        echo "[INFO] Copying artifacts from the 'orbit-$container_profile' container..."
        echo -e "\t - /workspace/orbit/logs -> ${SCRIPT_DIR}/artifacts/logs"
        echo -e "\t - /workspace/orbit/docs/_build -> ${SCRIPT_DIR}/artifacts/docs/_build"
        echo -e "\t - /workspace/orbit/data_storage -> ${SCRIPT_DIR}/artifacts/data_storage"
        # enter the script directory
        pushd ${SCRIPT_DIR} > /dev/null 2>&1
        # We have to remove before copying because repeated copying without deletion
        # causes strange errors such as nested _build directories
        # warn the user
        echo -e "[WARN] Removing the existing artifacts...\n"
        rm -rf ./artifacts/logs ./artifacts/docs/_build ./artifacts/data_storage

        # create the directories
        mkdir -p ./artifacts/docs

        # copy the artifacts
        docker cp orbit-$container_profile:/workspace/orbit/logs ./artifacts/logs
        docker cp orbit-$container_profile:/workspace/orbit/docs/_build ./artifacts/docs/_build
        docker cp orbit-$container_profile:/workspace/orbit/data_storage ./artifacts/data_storage
        echo -e "\n[INFO] Finished copying the artifacts from the container."
        popd > /dev/null 2>&1
        ;;
    stop)
        # Check that desired container is running, exit if it isn't
        is_container_running orbit-$container_profile
        echo "[INFO] Stopping the launched docker container orbit-$container_profile..."
        pushd ${SCRIPT_DIR} > /dev/null 2>&1
        docker compose --file docker-compose.yaml $add_profiles $add_envs down
        popd > /dev/null 2>&1
        ;;
    push)
        if ! command -v apptainer &> /dev/null; then
            install_apptainer
        fi
        # Check if Docker image exists
        check_image_exists orbit-$container_profile:latest
        # Check if Docker version is greater than 25
        check_docker_version
        # source env file to get cluster login and path information
        source $SCRIPT_DIR/.env.base
        # make sure exports directory exists
        mkdir -p /$SCRIPT_DIR/exports
        # clear old exports for selected profile
        rm -rf /$SCRIPT_DIR/exports/orbit-$container_profile*
        # create singularity image
        # NOTE: we create the singularity image as non-root user to allow for more flexibility. If this causes
        # issues, remove the --fakeroot flag and open an issue on the orbit repository.
        cd /$SCRIPT_DIR/exports
        APPTAINER_NOHTTPS=1 apptainer build --sandbox --fakeroot orbit-$container_profile.sif docker-daemon://orbit-$container_profile:latest
        # tar image (faster to send single file as opposed to directory with many files)
        tar -cvf /$SCRIPT_DIR/exports/orbit-$container_profile.tar orbit-$container_profile.sif
        # make sure target directory exists
        ssh $CLUSTER_LOGIN "mkdir -p $CLUSTER_SIF_PATH"
        # send image to cluster
        scp $SCRIPT_DIR/exports/orbit-$container_profile.tar $CLUSTER_LOGIN:$CLUSTER_SIF_PATH/orbit-$container_profile.tar
        ;;
    job)
        source $SCRIPT_DIR/.env.base
        # Check if singularity image exists on the remote host
        check_singularity_image_exists orbit-$container_profile
        # make sure target directory exists
        ssh $CLUSTER_LOGIN "mkdir -p $CLUSTER_ORBIT_DIR"
        # Sync orbit code
        echo "[INFO] Syncing orbit code..."
        rsync -rh  --exclude="*.git*" --filter=':- .dockerignore'  /$SCRIPT_DIR/.. $CLUSTER_LOGIN:$CLUSTER_ORBIT_DIR
        # execute job script
        echo "[INFO] Executing job script..."
        # check whether the second argument is a profile or a job argument
        if [ "$profile_arg" == "$container_profile" ] ; then
            # if the second argument is a profile, we have to shift the arguments
            echo "[INFO] Arguments passed to job script ${@:3}"
            ssh $CLUSTER_LOGIN "cd $CLUSTER_ORBIT_DIR && sbatch $CLUSTER_ORBIT_DIR/docker/cluster/submit_job.sh" "$CLUSTER_ORBIT_DIR" "orbit-$container_profile" "${@:3}"
        else
            # if the second argument is a job argument, we have to shift only one argument
            echo "[INFO] Arguments passed to job script ${@:2}"
            ssh $CLUSTER_LOGIN "cd $CLUSTER_ORBIT_DIR && sbatch $CLUSTER_ORBIT_DIR/docker/cluster/submit_job.sh" "$CLUSTER_ORBIT_DIR" "orbit-$container_profile" "${@:2}"
        fi
        ;;
    *)
        # Not recognized mode
        echo "[Error] Invalid command provided: $mode"
        print_help
        exit 1
        ;;
esac
