#!/usr/bin/env bash

# In case you need to load specific modules on the cluster, add them here
# e.g., `module load eth_proxy` or `ml go-1.19.4/apptainer-1.1.8`

scheduler="$1"
cluster_isaaclab_dir="$2"
container_profile="$3"

if [ "$scheduler" == "SLURM" ]; then
  cat <<'EOT' >> job.sh
#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=rtx_3090:1
#SBATCH --time=23:00:00
#SBATCH --mem-per-cpu=4048
#SBATCH --mail-type=END
#SBATCH --mail-user=name@mail
#SBATCH --job-name=training-$(date +"%Y-%m-%dT%H:%M")
EOT
elif [ "$scheduler" == "PBS" ]; then
  cat <<'EOT' >> job.sh
#!/bin/bash

#PBS -l select=1:ncpus=8:mpiprocs=1:ngpus=1
#PBS -l walltime=01:00:00
#PBS -j oe
#PBS -q gpu
#PBS -N isaaclab
#PBS -m bea -M "user@mail"
EOT
fi

cat <<EOT >> job.sh

# Pass the container profile first to run_singularity.sh, then all arguments intended for the executed script
sh "$cluster_isaaclab_dir/docker/cluster/run_singularity.sh" "$container_profile" "${@:4}"
EOT

# Submit the job
if [ "$scheduler" == "SLURM" ]; then
  sbatch < job.sh
elif [ "$scheduler" == "PBS" ]; then
  qsub job.sh
else
  echo "Invalid job scheduler. Available options [SLURM/PBS]."
  rm job.sh
  exit 1
fi

# Clean up the job script after submission
rm job.sh
