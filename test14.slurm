#!/bin/bash

#=========================================================
# SLURM Job Script for Parallel Hyperparameter Sweep
#=========================================================

# Name of the job
#SBATCH --job-name=bnn_sweep

# Output log files (%j will be replaced by the SLURM job ID)
#SBATCH --output=logs/sweep_%j.out
#SBATCH --error=logs/sweep_%j.err

# Resource Requests
#SBATCH --nodes=1                  # Run all processes on a single node
#SBATCH --ntasks=1                 # Run a single task (the python script)
#SBATCH --cpus-per-task=4         # Request 32 CPU cores for the multiprocessing pool
#SBATCH --mem=64G                  # Request 32 GB of total RAM
#SBATCH --time=12:00:00            # Time limit (HH:MM:SS)

#=========================================================
# Environment Setup & Execution
#=========================================================

module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
conda deactivate
conda activate torch28


echo "Allocated CPUs: $SLURM_CPUS_PER_TASK"
echo "Starting Python execution..."

python test14.py

my-job-stats -a -n -s

echo "Job completed successfully."