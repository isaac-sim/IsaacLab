#!/bin/bash

# Step 1: Grok the Ray address and set the RAY_ADDRESS environment variable
head_pod=$(kubectl get pods --no-headers -o custom-columns=":metadata.name" | grep 'head')

# Extract the RAY_ADDRESS from the logs of the Ray head pod
ray_address=$(kubectl logs $head_pod -c ray-head | grep -oP 'http://\d+\.\d+\.\d+\.\d+:\d+' | awk '{print $1}')

# Check if the RAY_ADDRESS is successfully retrieved
if [ -n "$ray_address" ]; then
    echo "$ray_address" > ~/.ray_address  # Store only the address in the file
    export RAY_ADDRESS=$ray_address
    echo "RAY_ADDRESS is set to: $RAY_ADDRESS"
else
    echo "Failed to retrieve RAY_ADDRESS."
    echo "Check your pods with kubectl get pods"
    echo "If your pods are running, inspect them with kubectl logs <POD>"
    echo "The Ray address should be in the logs."
    exit 1
fi
