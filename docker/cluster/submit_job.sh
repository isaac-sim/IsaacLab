#!/usr/bin/env bash

# In case you need to load specific modules on the cluster, add them here
# e.g., `module load eth_proxy` or `ml go-1.19.4/apptainer-1.1.8`

# Check for the required number of arguments
if [ "$#" -lt 4 ]; then
  echo "Usage: $0 <SLURM|PBS> <CLUSTER_ISAACLAB_DIR> <CONTAINER_PROFILE> <additional_args...>"
  exit 1
fi

scheduler="$1"
isaaclab_dir="$2"
container_profile="$3"

cat <<EOT > job.sh
#!/bin/bash

$(if [ "$scheduler" == "SLURM" ]; then
  echo "#SBATCH -n 1"
  echo "#SBATCH --cpus-per-task=8"
  echo "#SBATCH --gpus=rtx_3090:1"
  echo "#SBATCH --time=23:00:00"
  echo "#SBATCH --mem-per-cpu=4048"
  echo "#SBATCH --mail-type=END"
  echo "#SBATCH --mail-user=name@mail"
  echo "#SBATCH --job-name=training-$(date +"%Y-%m-%dT%H:%M")"
elif [ "$scheduler" == "PBS" ]; then
  echo "#PBS -l select=1:ncpus=2:mpiprocs=12:ngpus=1"
  echo "#PBS -l walltime=01:00:00"
  echo "#PBS -j oe"
  echo "#PBS -q gpu"
  echo "#PBS -N isaaclab"
fi)

# Pass the container profile first to run_singularity.sh, then all arguments intended for the executed script
sh "$isaaclab_dir/docker/cluster/run_singularity.sh" "$container_profile" "${@:4}"
EOT

# Submit the job
if [ "$scheduler" == "SLURM" ]; then
  sbatch < job.sh
elif [ "$scheduler" == "PBS" ]; then
  qsub job.sh
else
  echo "Invalid argument. Please specify 'SLURM' or 'PBS'."
  rm job.sh
  exit 1
fi

# Clean up the job script after submission
rm job.sh
