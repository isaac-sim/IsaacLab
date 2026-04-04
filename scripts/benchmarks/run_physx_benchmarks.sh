#!/usr/bin/env bash

# Set output directory from argument or use default
ROOT_DIR="${1:-./benchmarks}"
OUTPUT_DIR="${ROOT_DIR}/isaaclab_physx"

./isaaclab.sh -p source/isaaclab_physx/benchmark/assets/benchmark_articulation.py --backend omniperf --output_dir "$OUTPUT_DIR"
./isaaclab.sh -p source/isaaclab_physx/benchmark/assets/benchmark_rigid_object.py --backend omniperf --output_dir "$OUTPUT_DIR"
./isaaclab.sh -p source/isaaclab_physx/benchmark/assets/benchmark_rigid_object_collection.py --backend omniperf --output_dir "$OUTPUT_DIR"
./isaaclab.sh -p source/isaaclab_physx/benchmark/assets/benchmark_articulation_data.py --backend omniperf --output_dir "$OUTPUT_DIR"
./isaaclab.sh -p source/isaaclab_physx/benchmark/assets/benchmark_rigid_object_data.py --backend omniperf --output_dir "$OUTPUT_DIR"
./isaaclab.sh -p source/isaaclab_physx/benchmark/assets/benchmark_rigid_object_collection_data.py --backend omniperf --output_dir "$OUTPUT_DIR"
