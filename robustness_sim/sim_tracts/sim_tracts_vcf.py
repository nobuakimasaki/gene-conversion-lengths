### This file is used to simulate gene conversion tracts on individuals from the coalescent simulation.
### Gene conversion tracts are drawn from a geometric distribution.

import allel
import time
import numpy as np
import pandas as pd
import csv
import multiprocessing
from functools import partial
import random
import sys

random.seed(27)
np.random.seed(27)

print("n cores:")
print(multiprocessing.cpu_count())

def sim_tracts(N_gene_conv, min_pos, max_pos, dist_N, mean=100):
    tracts = []
    while len(tracts) < N_gene_conv:
        # Generate start positions
        start = np.random.choice(range(min_pos, max_pos-2000), N_gene_conv, replace=True)

        # Generate lengths based on the specified distribution
        if dist_N == "geom":
            lengths = np.random.geometric(1/mean, N_gene_conv)
        elif dist_N == "geom2":
            lengths = np.random.geometric(2/mean, N_gene_conv) + np.random.geometric(2/mean, N_gene_conv)
        elif dist_N == "unif":
            lengths = np.random.uniform(low=1, high=mean*2-1, size=N_gene_conv)
        elif dist_N == "geom3":
            lengths = np.random.geometric(3/mean, N_gene_conv) + np.random.geometric(3/mean, N_gene_conv) + np.random.geometric(3/mean, N_gene_conv)
        elif dist_N == "mixture":
            w1 = 0.05
            mu1 = 700
            mu2 = (mean - w1 * mu1) / (1 - w1)
            p1 = 1 / mu1
            p2 = 1 / mu2
            choice = np.random.rand(N_gene_conv) < w1
            lengths = np.where(choice, np.random.geometric(p1, N_gene_conv), np.random.geometric(p2, N_gene_conv))
        else:
            raise ValueError("Invalid distribution type.")

        # Calculate end positions
        end = start + lengths - 1
        start_end = np.column_stack((start, end))

        # Filter out rows where the end position is greater than max_pos
        valid_tracts = start_end[start_end[:, 1] < max_pos]

        # Append the valid tracts
        tracts.extend(valid_tracts.tolist())

    # After while loop, trim to exactly N_gene_conv
    tracts = tracts[:N_gene_conv]
    return tracts

def sim_gene_conv(actual, idx, positions, genotypes):

    geno_series = genotypes[:, idx]
    # Convert geno_series to a NumPy array for easier manipulation
    geno_array = np.array(geno_series)
    # Create a new list with 1 where geno_series element is [0, 0] and 0 otherwise
    heterozygous = [1 if np.array_equal(geno, [1, 0]) or np.array_equal(geno, [0, 1]) else 0 for geno in geno_array]
    heterozygous = np.array(heterozygous)

    # print("heterozygous: ")
    # print(heterozygous)

    # Define the start and end of the actual tract
    start, end = actual
    # print("start: ")
    # print(start)
    # print("end: ")
    # print(end)
    # Identify the indices of the positions within the tract
    tract_ind = np.where((positions >= start) & (positions <= end))[0]
    tract_ind = tract_ind.astype(int)
    print("tract_ind: ")
    print(tract_ind)

    # If there are no heterozygous sites within tract
    if len(tract_ind) == 0:
        # print("L: ")
        # print(0)
        return [0, 0, 0]

    # Extract the heterozygous markers within the tract
    tract = heterozygous[tract_ind]

    # Calculate the length of the gene conversion tract

    # If no heterozygous markers
    if np.sum(tract) == 0:
        result = [0, 0, 0]
    # If one heterozygous marker
    elif np.sum(tract) == 1:
        obs_start = positions[tract_ind[np.min(np.where(tract == 1))]]
        obs_end = positions[tract_ind[np.max(np.where(tract == 1))]]
        L = obs_end - obs_start + 1
        result = [obs_start, obs_end, L]
    # If more than one heterozygous marker
    else:
        # Get the index of the leftmost and rightmost markers and locate the positions
        obs_start = positions[tract_ind[np.min(np.where(tract == 1))]]
        obs_end = positions[tract_ind[np.max(np.where(tract == 1))]]
        L = obs_end - obs_start + 1
        result = [obs_start, obs_end, L]

    print("result: ")
    print(result)
    return result

# Define the file path
file_path = "/projects/browning/brwnlab/sharon/for_nobu/gc_length/sim5_data/sim5_seed1_10Mb_n125000.gtstats"
# Read the table into a pandas DataFrame
maf_df = pd.read_table(file_path, header=None)  # Assuming the file has no header
# Filter the DataFrame and select column V2 where V11 < 0.05
keep = maf_df[maf_df.iloc[:, 10] >= 0.05].iloc[:, 1].astype(int).tolist()
print("keep: ")
print(keep)

###### So far, we've defined the functions and the MAF file that will be used to generate the tracts.
###### We next load in the genotypes for individuals and actually simulate the tracts.

# Path to your compressed VCF file
vcf_file = "/projects/browning/brwnlab/sharon/for_nobu/gc_length/sim5_vcfs/sim5_seed1_10Mb_n125000_err0.0002phased_del1.vcf.gz"
# vcf_file = "example.vcf"
# Open the compressed VCF file for reading
callset = allel.read_vcf(vcf_file)
print("callset: ")
print(sorted(callset.keys()))
# Extract the samples, genotype calls, and variant positions
samples = callset['samples']
genotype_calls = callset['calldata/GT']
variant_positions = callset['variants/POS']
variant_positions = variant_positions.astype(int)

filtered_indices = [pos in keep for pos in variant_positions]
filtered_positions = variant_positions[filtered_indices]
filtered_genotypes = genotype_calls[filtered_indices]

print("filtered_genotypes: ")
print(filtered_genotypes[:5, :5])

# Specify the number of individuals you want to sample
N = 10**5
num_iterations = 100  # Specify the number of iterations

all_data = []  # List to store the concatenated results

# Get distribution from command line
if len(sys.argv) < 2:
    print("Usage: python script.py <distribution>")
    print("Available: geom, geom2, geom3, unif, mixture")
    sys.exit(1)

dist = sys.argv[1]
if dist not in ["geom", "geom2", "geom3", "unif", "mixture"]:
    raise ValueError(f"Invalid distribution: {dist}")

print(f"\n==== Simulating for distribution: {dist} ====")

all_data = []  # Reset for each distribution

for iteration in range(num_iterations):
    print(f"Iteration {iteration} for {dist}")
    
    # Simulate gene conversion tracts for this distribution
    gene_conversion_tracts = sim_tracts(N, min(filtered_positions), max(filtered_positions), dist)

    fixed_positions_sim_gene_conv = partial(sim_gene_conv, positions=filtered_positions, genotypes=filtered_genotypes)

    data = []
    for i in range(N):
        idx = np.random.choice(filtered_genotypes.shape[1])
        actual = gene_conversion_tracts[i]
        data.append((actual, idx))

    with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1) as pool:
        L = pool.starmap(fixed_positions_sim_gene_conv, data)

    # Add iteration and distribution to each result
    L_with_info = [lst + [iteration, dist] for lst in L if lst[-1] != 0]
    all_data.extend(L_with_info)

# Write each distribution's results to its own file
output_file = f"sim_tracts_vcf_{dist}_multiple_iterations.csv"
with open(output_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["obs_start", "obs_end", "length", "iteration", "distribution"])
    writer.writerows(all_data)

print(f"Finished writing {output_file}")