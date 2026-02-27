#!/usr/bin/env bash

# in the case you need to load specific modules on the cluster, add them here
# e.g., `module load eth_proxy`

# create job script with compute demands
### MODIFY HERE FOR YOUR JOB ###
cat <<EOT > job.sh
#!/bin/bash

#PBS -l select=1:ncpus=8:mpiprocs=1:ngpus=1
#PBS -l walltime=01:00:00
#PBS -j oe
#PBS -q gpu
#PBS -N isaaclab
#PBS -m bea -M "user@mail"

# Pass the container profile first to run_singularity.sh, then all arguments intended for the executed script
bash "$1/docker/cluster/run_singularity.sh" "$1" "$2" "${@:3}"
EOT

qsub job.sh
rm job.sh
