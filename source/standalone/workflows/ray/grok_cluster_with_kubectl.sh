#!/bin/bash

# Default cluster name prefix if no argument is provided
CLUSTER_NAME_PREFIX="isaac-lab-hyperparameter-tuner"
CLUSTER_SPEC_FILE="$HOME/.cluster_spec"  # Use absolute path

# Allow optional name argument for the cluster name prefix
if [ $# -gt 0 ]; then
    CLUSTER_NAME_PREFIX="$1"
fi

# Ensure the cluster spec file exists, if not, create it
if [ ! -f "$CLUSTER_SPEC_FILE" ]; then
    touch "$CLUSTER_SPEC_FILE"
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
    RAY_ADDRESS=$2
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
        POD_STATUS=$(kubectl get pod "$POD" -o jsonpath='{.status.phase}')
        if [[ "$POD_STATUS" != "Running" ]]; then
            PENDING_PODS_WARNING=true
            continue
        fi

        GPUS=$(kubectl get pod "$POD" -o jsonpath='{.spec.containers[0].resources.limits.nvidia\.com/gpu}')
        TOTAL_GPUS=$(($TOTAL_GPUS + ${GPUS:-0}))

        CPUS=$(kubectl get pod "$POD" -o jsonpath='{.spec.containers[0].resources.limits.cpu}')
        if [[ $CPUS == *m ]]; then
            CPUS=$(echo "scale=2; $CPUS / 1000" | bc)
        fi
        TOTAL_CPUS=$(echo "$TOTAL_CPUS + ${CPUS:-0}" | bc)

        RAM_MIB=$(kubectl exec -it "$POD" -c ray-worker -- bash -c "free -m | grep Mem | awk '{print \$2}'" | tr -d '\r')
        RAM_GIB=$(convert_memory_mib_to_gib "${RAM_MIB:-0}")
        TOTAL_RAM=$(echo "$TOTAL_RAM + $RAM_GIB" | bc)
    done

    SINGLE_LINE_OUTPUT="name: $CLUSTER_NAME address: ${RAY_ADDRESS} num_cpu: $TOTAL_CPUS num_gpu: $TOTAL_GPUS ram_gb: $TOTAL_RAM"
    echo "$SINGLE_LINE_OUTPUT" >> "$CLUSTER_SPEC_FILE"
}

# Get all cluster names
CLUSTERS=$(kubectl get pods | grep -oP "${CLUSTER_NAME_PREFIX}-\d+" | sort | uniq)

if [ -z "$CLUSTERS" ]; then
    echo "No clusters found with prefix ${CLUSTER_NAME_PREFIX}."
    exit 1
fi

# Clear the output file before appending new data
> "$CLUSTER_SPEC_FILE"

# Extract the RAY_ADDRESS from each head pod's logs and get the cluster name
head_pods=$(kubectl get pods --no-headers -o custom-columns=":metadata.name" | grep 'head')

for head_pod in $head_pods; do
    cluster_name=$(echo "$head_pod" | sed 's/-head.*//')

    # Get the logs for this pod, defaulting to the ray-head container
    pod_logs=$(kubectl logs "$head_pod" -c ray-head)

    # Extract the RAY_ADDRESS
    ray_address=$(echo "$pod_logs" | grep -oP "RAY_ADDRESS='http://[^']*'" | sed "s/RAY_ADDRESS='//;s/'//")

    if [ -z "$ray_address" ]; then
        echo "Error: Could not find RAY_ADDRESS for cluster $cluster_name" >> "$CLUSTER_SPEC_FILE"
        continue
    fi

    get_gpu_worker_resources "$cluster_name" "$ray_address"
done

# Print the contents of the cluster spec file
echo "Cluster spec information saved to $CLUSTER_SPEC_FILE"
cat "$CLUSTER_SPEC_FILE"
