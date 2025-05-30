#!/bin/bash
#SBATCH --job-name=UK_mixture         # Job name
#SBATCH --output=output_%A_%a.log         # Standard output (%A=array job ID, %a=array index)
#SBATCH --error=error_%A_%a.log           # Standard error
#SBATCH --partition=b-students            # Partition name
#SBATCH --ntasks=1                        # Single task
#SBATCH --cpus-per-task=13                # Number of CPU cores
#SBATCH --mem=4G                          # Memory
#SBATCH --array=1-42%15                   # Job array for seeds 1 to 25
#SBATCH --mail-type=END,FAIL              # Email on job end/fail
#SBATCH --mail-user=masakin@uw.edu        # Your email

# Activate your virtual environment
source ../myenv/bin/activate

# Run Python script with array index as seed argument
python3 fit.model.py ${SLURM_ARRAY_TASK_ID}
