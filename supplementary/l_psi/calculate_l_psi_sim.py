import sys
import os
import numpy as np
import pandas as pd

# Function to load observed tract lengths
def read_tracts(seed, MAF):
    base = "../../simulation/data/"
    suffix = {0.5: "1.0", 0.4: "0.4", 0.3: "0.3"}.get(MAF)
    if suffix is None:
        raise ValueError("Unsupported MAF value")
    filename = f"sim5_seed{seed}_10Mb_n125000_ibdclust2cM_trim1_maf0.1_gcmaf0.05_9kbgaps_combinedoffsets_err0.0002_del1_{suffix}_421.inf_obs_tracts"
    return pd.read_csv(base + filename, sep=r'\s+', header=None, engine='python')

# Function to load MAF file
def read_MAF(seed):
    return pd.read_table(f"../../simulation/data/sim5_seed{seed}_10Mb_n125000.gtstats", header=None)

# Estimate psi for one tract
def est_psi(observed, psi, region, length_chrom):
    left = max(int(observed[0]) - region, 0)
    right = min(int(observed[1]) + region, length_chrom - 1)
    return np.mean(psi[left:right+1])

# Main function for processing one seed
def get_l_psi(MAF_df, df, M, region, MAF_ceil):
    max_pos = MAF_df[1].max()
    psi = np.zeros(max_pos + 1)

    # Compute psi values
    maf_values = MAF_df[10]
    positions = MAF_df[1]
    psi[positions] = 2 * maf_values * (1 - maf_values)

    # Set psi to 0 where MAF is < 0.05 or > MAF_ceil
    exc = MAF_df[(MAF_df[10] < 0.05) | (MAF_df[10] > MAF_ceil)]
    psi[exc[1]] = 0

    # Filter out singleton tracts and long tracts
    tract_lengths = df[1] - df[0] + 1
    keep = (tract_lengths <= M)
    df = df[keep]

    print("num tracts:")
    print(len(df))

    # Generate psi and length lists
    tracts_lst = [row for _, row in df.iterrows()]
    psi_lst = [est_psi(row, psi=psi, region=region, length_chrom=len(psi))
               for row in tracts_lst]

    print("psi: ")
    print(psi_lst[:10])

    l_lst = [(row[1] - row[0] + 1) for row in tracts_lst]

    print("l: ")
    print(l_lst[:10])

    print("finished processing data for 1 chromosome")

    return l_lst, psi_lst

# Process all 20 seeds and save combined l and psi
if __name__ == "__main__":
    all_l = []
    all_psi = []

    for seed in range(1, 21):
        print(f"Processing seed {seed}")
        tracts_df = read_tracts(seed, MAF=0.5)
        maf_df = read_MAF(seed)
        l, psi = get_l_psi(maf_df, tracts_df, M=1500, region=5000, MAF_ceil=0.5)
        all_l.extend(l)
        all_psi.extend(psi)

    df = pd.DataFrame({"l": all_l, "psi": all_psi})
    df.to_csv("l_psi_sim.csv", index=False)