#!/bin/bash

# Step 1: Get a list of all head pods and their IPs
head_pods=$(kubectl get pods --no-headers -o custom-columns=":metadata.name" | grep 'head')

# Count how many head pods were found
num_heads=$(echo "$head_pods" | wc -l)

# Notify the user how many head pods were found
echo "$num_heads Ray head pod(s) found."

# Initialize an empty file to store Ray addresses and their cluster names
> ~/.ray_address

# Step 2: Extract the RAY_ADDRESS (IP and port) from the logs of each head pod and get the cluster name
for head_pod in $head_pods; do
    # Get the cluster name (part of the head pod's name before "-head")
    cluster_name=$(echo "$head_pod" | sed 's/-head.*//')

    # Get the IP of the head pod
    pod_ip=$(kubectl get pod "$head_pod" -o jsonpath='{.status.podIP}')

    # The Ray cluster typically runs on port 6379, include the port with the IP
    ray_port="6379"
    ray_address="$pod_ip:$ray_port"

    # Print and write the results to the file with comments above the IP
    echo "# Cluster: $cluster_name" >> ~/.ray_address
    echo "$ray_address" >> ~/.ray_address
done

# Step 3: Final check to see if addresses were written to the file
if [ -s ~/.ray_address ]; then
    echo "All Ray IPs and cluster names were written to ~/.ray_address:"
    echo "<--- Contents of file below"
    cat ~/.ray_address
    echo "<--- Contents of file above"
else
    echo "No Ray IP addresses were found."
    exit 1
fi
