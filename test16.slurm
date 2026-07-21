#!/bin/bash

#=========================================================
# SLURM Job Script for M_ARTIFICIAL_CHANNELS Sweep
#=========================================================

# Name of the job
#SBATCH --job-name=bnn_sweep_m

# Output log files (%j will be replaced by the SLURM job ID)
#SBATCH --output=logs/sweep_m_%j.out
#SBATCH --error=logs/sweep_m_%j.err

# Resource Requests
#SBATCH --nodes=1                  # Run all processes on a single node
#SBATCH --ntasks=1                 # Run a single task (the python script)
#SBATCH --cpus-per-task=64         # Request CPU cores for the multiprocessing pool
#SBATCH --mem=1024G                # Request RAM
#SBATCH --time=24:00:00            # Time limit (HH:MM:SS)

#=========================================================
# Environment Setup & Execution
#=========================================================

module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
conda deactivate
conda activate torch28

echo "Allocated CPUs: $SLURM_CPUS_PER_TASK"
echo "Starting Python execution for test16.py..."

python test16.py

my-job-stats -a -n -s

echo "Job completed successfully."
