#!/bin/bash

# Default cluster name prefix if no argument is provided
CLUSTER_NAME_PREFIX="isaac-lab-hyperparameter-tuner"
RAY_PORT="6379"
CLUSTER_SPEC_FILE="$HOME/.cluster_spec"  # Use absolute path

# Allow optional name argument for the cluster name prefix
if [ $# -gt 0 ]; then
    CLUSTER_NAME_PREFIX="$1"
fi

# Print a message indicating the cluster prefix being filtered
echo "Filtering for cluster name prefix: $CLUSTER_NAME_PREFIX"
echo "Change your cluster name prefix by supplying it as a script argument like ./capture_cluster_spec.sh <PREFIX>"
echo "<-------------------------------------------------------------"

# Ensure the cluster spec file exists, if not, create it
if [ ! -f "$CLUSTER_SPEC_FILE" ]; then
    touch "$CLUSTER_SPEC_FILE"
    echo "Created cluster spec file at $CLUSTER_SPEC_FILE."
fi

# Function to convert memory from MiB to GiB
convert_memory_mib_to_gib() {
    RAM_MIB=$1
    RAM_GIB=$(echo "scale=2; ${RAM_MIB}/1024" | bc)
    echo $RAM_GIB
}

# Function to get total resources (GPU, CPU, RAM) for GPU worker nodes in a specific cluster
get_gpu_worker_resources() {
    CLUSTER_NAME=$1
    POD_IP=$2
    PENDING_PODS_WARNING=false

    # Get the worker pods for the cluster
    WORKER_PODS=$(kubectl get pods | grep "${CLUSTER_NAME}-worker" | awk '{print $1}')

    if [ -z "$WORKER_PODS" ]; then
        echo "Warning: No worker pods found for cluster $CLUSTER_NAME." >> "$CLUSTER_SPEC_FILE"
        return
    fi

    # Initialize totals
    TOTAL_GPUS=0
    TOTAL_CPUS=0
    TOTAL_RAM=0

    # Iterate over each worker pod and sum the resources, only for GPU workers
    for POD in $WORKER_PODS; do
        # Check the pod's status
        POD_STATUS=$(kubectl get pod "$POD" -o jsonpath='{.status.phase}')
        if [[ "$POD_STATUS" != "Running" ]]; then
            echo "Warning: Pod $POD is in $POD_STATUS state, skipping..."
            PENDING_PODS_WARNING=true
            continue
        fi

        # Get the number of GPUs for the worker pod
        GPUS=$(kubectl get pod "$POD" -o jsonpath='{.spec.containers[0].resources.limits.nvidia\.com/gpu}')
        if [ -z "$GPUS" ] || [ "$GPUS" -eq 0 ]; then
            continue
        fi
        TOTAL_GPUS=$(($TOTAL_GPUS + $GPUS))

        # Get the number of CPUs (and handle fractional CPUs)
        CPUS=$(kubectl get pod "$POD" -o jsonpath='{.spec.containers[0].resources.limits.cpu}')
        if [ -z "$CPUS" ]; then
            CPUS=0
        fi
        if [[ $CPUS == *m ]]; then
            CPUS=$(echo "scale=2; $CPUS / 1000" | bc)
        fi
        TOTAL_CPUS=$(echo "$TOTAL_CPUS + $CPUS" | bc)

        # Get RAM using `kubectl exec` to run `free -m` inside the pod and specify the container
        RAM_MIB=$(kubectl exec -it "$POD" -c ray-worker -- bash -c "free -m | grep Mem | awk '{print \$2}'" | tr -d '\r')
        if [ -z "$RAM_MIB" ]; then
            RAM_MIB=0
        fi
        RAM_GIB=$(convert_memory_mib_to_gib "$RAM_MIB")
        TOTAL_RAM=$(echo "$TOTAL_RAM + $RAM_GIB" | bc)
    done

    # Print and append the total resources for GPU workers in the cluster in a single line
    SINGLE_LINE_OUTPUT="name: $CLUSTER_NAME address: ${POD_IP}:${RAY_PORT} num_cpu: $TOTAL_CPUS num_gpu: $TOTAL_GPUS ram_gb: $TOTAL_RAM"
    echo "$SINGLE_LINE_OUTPUT"
    echo "$SINGLE_LINE_OUTPUT" >> "$CLUSTER_SPEC_FILE"

    # Display warning if any pods were skipped
    if [ "$PENDING_PODS_WARNING" = true ]; then
        echo "Warning: Some pods in $CLUSTER_NAME were not running and were skipped." >> "$CLUSTER_SPEC_FILE"
    fi
}

# Get all cluster names (assuming clusters follow the naming pattern provided by the user or the default)
CLUSTERS=$(kubectl get pods | grep -oP "${CLUSTER_NAME_PREFIX}-\d+" | sort | uniq)

if [ -z "$CLUSTERS" ]; then
    echo "No clusters found with prefix ${CLUSTER_NAME_PREFIX}."
    exit 1
fi

# Clear the output file before appending new data
> "$CLUSTER_SPEC_FILE"

# Step 1: Get a list of all head pods and their IPs
head_pods=$(kubectl get pods --no-headers -o custom-columns=":metadata.name" | grep 'head')

if [ -z "$head_pods" ]; then
    echo "No Ray head pods found."
    exit 1
fi

# Step 2: Extract the RAY_ADDRESS (IP and port) from each head pod and get the cluster name
for head_pod in $head_pods; do
    # Get the cluster name (part of the head pod's name before "-head")
    cluster_name=$(echo "$head_pod" | sed 's/-head.*//')

    # Get the IP of the head pod
    pod_ip=$(kubectl get pod "$head_pod" -o jsonpath='{.status.podIP}')

    # Now gather and display the resource info for the GPU worker nodes of this cluster
    get_gpu_worker_resources "$cluster_name" "$pod_ip"
done

echo "Cluster spec information saved to $CLUSTER_SPEC_FILE"

