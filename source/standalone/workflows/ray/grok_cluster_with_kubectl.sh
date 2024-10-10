#!/bin/bash

# Default values for the cluster name prefix and result file path
CLUSTER_NAME_PREFIX="isaac-lab-hyperparameter-tuner"
CLUSTER_SPEC_FILE="$HOME/.cluster_spec"  # Default absolute path
echo "./grok_cluster_with_kubectl.sh --name CLUSTER_PREFIX --result_file RESULT_FILE"

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --name) CLUSTER_NAME_PREFIX="$2"; shift ;;
        --result_file) CLUSTER_SPEC_FILE="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Ensure the result file exists or create it
if [ ! -f "$CLUSTER_SPEC_FILE" ]; then
    touch "$CLUSTER_SPEC_FILE"
fi

# Function to check if all clusters are running
check_clusters_running() {
    all_clusters_running=true
    for cluster_name in $CLUSTERS; do
        # Check if the head pod is running for each cluster
        HEAD_POD=$(kubectl get pods | grep "${cluster_name}-head" | awk '{print $1}')
        HEAD_POD_STATUS=$(kubectl get pods "$HEAD_POD" -o=jsonpath='{.status.phase}')

        if [ "$HEAD_POD_STATUS" != "Running" ]; then
            all_clusters_running=false
            break
        fi
    done
    echo $all_clusters_running
}

# Function to get total resources (GPU, CPU, RAM) and workers for GPU worker nodes in a specific cluster
get_gpu_worker_resources() {
    CLUSTER_NAME=$1
    RAY_ADDRESS=$2

    # Use kubectl get pods to count the number of worker pods
    WORKER_PODS=$(kubectl get pods | grep "${CLUSTER_NAME}-worker-gpu-group" | wc -l)

    # Get the Ray status for this cluster
    ray_status=$(kubectl exec -it "$HEAD_POD" -c ray-head -- ray status 2>/dev/null)

    # Parse the ray status output for resource information
    TOTAL_CPUS=$(echo "$ray_status" | grep -oP '([0-9.]+)\/[0-9.]+ CPU' | cut -d '/' -f2 | cut -d ' ' -f1)
    TOTAL_GPUS=$(echo "$ray_status" | grep -oP '([0-9.]+)\/[0-9.]+ GPU' | cut -d '/' -f2 | cut -d ' ' -f1)

    # Extract total memory in GiB, remove the "GiB" suffix and capture only the first occurrence
    TOTAL_RAM=$(echo "$ray_status" | grep -oP '[0-9.]+GiB' | head -n 1 | sed 's/GiB//')

    # If TOTAL_RAM is blank or missing, set a default value
    if [ -z "$TOTAL_RAM" ]; then
        TOTAL_RAM="0"  # Default to 0 if parsing fails
    fi

    # Format the output line for the cluster spec file
    SINGLE_LINE_OUTPUT="name: $CLUSTER_NAME address: ${RAY_ADDRESS} num_cpu: $TOTAL_CPUS num_gpu: $TOTAL_GPUS ram_gb: $TOTAL_RAM total_workers: $WORKER_PODS"
    echo "$SINGLE_LINE_OUTPUT" >> "$CLUSTER_SPEC_FILE"
}

# Get all cluster names
CLUSTERS=$(kubectl get pods | grep -oP "${CLUSTER_NAME_PREFIX}-\d+" | sort | uniq)

if [ -z "$CLUSTERS" ]; then
    echo "No clusters found with prefix ${CLUSTER_NAME_PREFIX}."
    exit 1
fi

# Wait for all clusters to spin up
while true; do
    if [ "$(check_clusters_running)" == "true" ]; then
        break
    else
        echo "Waiting for all clusters to spin up..."
        sleep 5
    fi
done

# Clear the output file before appending new data
> "$CLUSTER_SPEC_FILE"

# Iterate over each cluster, find the head pod dynamically, run ray status, and get resources
for cluster_name in $CLUSTERS; do
    # Dynamically find the head pod name using kubectl
    HEAD_POD=$(kubectl get pods | grep "${cluster_name}-head" | awk '{print $1}')

    if [ -z "$HEAD_POD" ]; then
        echo "Error: Could not find head pod for cluster $cluster_name" >> "$CLUSTER_SPEC_FILE"
        continue
    fi

    # Get the logs for the head pod, defaulting to the ray-head container
    pod_logs=$(kubectl logs "$HEAD_POD" -c ray-head)

    # Extract the RAY_ADDRESS from the pod logs
    ray_address=$(echo "$pod_logs" | grep -oP "RAY_ADDRESS='http://[^']*'" | sed "s/RAY_ADDRESS='//;s/'//")

    if [ -z "$ray_address" ]; then
        echo "Error: Could not find RAY_ADDRESS for cluster $cluster_name" >> "$CLUSTER_SPEC_FILE"
        continue
    fi

    # Get the GPU, CPU, RAM, and worker information using ray status
    get_gpu_worker_resources "$cluster_name" "$ray_address"
done

# Print the contents of the cluster spec file
echo "Cluster spec information saved to $CLUSTER_SPEC_FILE"
cat "$CLUSTER_SPEC_FILE"
