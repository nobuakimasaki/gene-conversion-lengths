#!/bin/bash
#SBATCH --job-name=sim_tracts         # Job name
#SBATCH --output=output_%j.log        # Standard output (%j = job ID)
#SBATCH --error=error_%j.log          # Standard error
#SBATCH --partition=b-students        # Partition name
#SBATCH --ntasks=1                    # Single task
#SBATCH --cpus-per-task=1             # Number of CPU cores
#SBATCH --mem=4G                      # Memory
#SBATCH --mail-type=END,FAIL          # Email on job end/fail
#SBATCH --mail-user=masakin@uw.edu    # Your email

# Load any required modules or activate environment if needed
# module load R               

# Run your R script
Rscript sim_tracts.R
