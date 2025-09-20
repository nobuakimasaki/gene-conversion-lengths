#!/bin/bash
#SBATCH --job-name=fit_model
#SBATCH --output=output_%A_%a.log
#SBATCH --error=error_%A_%a.log
#SBATCH --partition=b-20c256g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=4G
#SBATCH --array=0-9
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=masakin@uw.edu

set -euo pipefail
source ../myenv/bin/activate

# Sweep settings
dists=(geom geom2 geom3 unif mixture)
means=(21 100)

# Map SLURM_ARRAY_TASK_ID -> (dist, mean)
dist_idx=$(( SLURM_ARRAY_TASK_ID % ${#dists[@]} ))
mean_idx=$(( SLURM_ARRAY_TASK_ID / ${#dists[@]} ))

dist=${dists[$dist_idx]}
mean=${means[$mean_idx]}

echo "[$(date)] Task ${SLURM_ARRAY_TASK_ID}: dist=${dist}, mean=${mean}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Run
python3 fit.model.py --dist "${dist}" --mean "${mean}"
