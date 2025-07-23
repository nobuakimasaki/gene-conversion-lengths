#!/bin/bash
#SBATCH --job-name=fit_model
#SBATCH --output=output_%A_%a.log
#SBATCH --error=error_%A_%a.log
#SBATCH --partition=b-students
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=4G
#SBATCH --array=0-4                   
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=masakin@uw.edu

source ../myenv/bin/activate

# Define array of distribution names
distributions=("geom" "geom2" "geom3" "unif" "mixture")

# Use SLURM_ARRAY_TASK_ID to select distribution
dist=${distributions[$SLURM_ARRAY_TASK_ID]}

echo "Running simulation for distribution: $dist"

python3 fit.model.py --dist $dist
