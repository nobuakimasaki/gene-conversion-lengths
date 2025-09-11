#!/bin/bash
#SBATCH --job-name=sim_tracts
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --partition=b-20c256g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=4G
#SBATCH --array=0-9            # optionally: 0-9%4 to cap concurrency at 4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=masakin@uw.edu

set -euo pipefail

mkdir -p logs
source ../myenv/bin/activate

# Decode (dist, scenario) from array index
dists=(geom geom2 geom3 unif mixture)
means=(21 100)
mix_mu1s=(725 700)
mix_w1s=(0.005 0.05)

dist_idx=$(( SLURM_ARRAY_TASK_ID % ${#dists[@]} ))   # 0..4
sc_idx=$(( SLURM_ARRAY_TASK_ID / ${#dists[@]} ))     # 0..1

dist=${dists[$dist_idx]}
mean=${means[$sc_idx]}
mu1=${mix_mu1s[$sc_idx]}
w1=${mix_w1s[$sc_idx]}

echo "[$(date)] task_id=${SLURM_ARRAY_TASK_ID} dist=${dist} | mean=${mean} | mix_mu1=${mu1} | mix_w1=${w1}"
echo "CPUs requested: ${SLURM_CPUS_PER_TASK}"

# Make numpy/numexpr respect cpus
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_MAX_THREADS=${SLURM_CPUS_PER_TASK}

python3 sim_tracts_vcf.py \
  --dist "${dist}" \
  --mean "${mean}" \
  --mix_mu1 "${mu1}" \
  --mix_w1 "${w1}" \
  --cpus "${SLURM_CPUS_PER_TASK}"

echo "[$(date)] Done."
