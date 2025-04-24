import sys
import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.optimize import dual_annealing
from itertools import chain
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import random

sys.path.append(os.path.abspath("../UK_biobank"))
import model

# Function to load observed tract lengths from one region
def read_tracts(seed, MAF):
    #base = "/projects/browning/brwnlab/sharon/for_nobu/gc_length/sim5_data/"
    base = "data/"
    if MAF == 0.5:
        suffix = "1.0"
    elif MAF == 0.4:
        suffix = "0.4"
    elif MAF == 0.3:
        suffix = "0.3"
    else:
        raise ValueError("Unsupported MAF value")
    filename = f"sim5_seed{seed}_10Mb_n125000_ibdclust2cM_trim1_maf0.1_gcmaf0.05_9kbgaps_combinedoffsets_err0.0002_del1_{suffix}_421.inf_obs_tracts"
    df = pd.read_csv(base + filename, sep=r'\s+', header=None, engine='python')
    return df

# Function to read MAF file from one region
def read_MAF(seed):
    #path = f"/projects/browning/brwnlab/sharon/for_nobu/gc_length/sim5_data/sim5_seed{seed}_10Mb_n125000.gtstats"
    path = f"data/sim5_seed{seed}_10Mb_n125000.gtstats"
    df = pd.read_table(path, header=None, index_col=False)
    return df

def est_psi(observed, psi, region, length_chrom):
    leftmost = max(int(observed[0]) - region, 0)
    rightmost = min(int(observed[1]) + region, length_chrom - 1)
    index = list(range(leftmost, rightmost + 1))
    return np.mean(psi[index])

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
    keep = (tract_lengths <= M) & (tract_lengths > 1)
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

def bootstrap_once(args):
    psi_lst_filt, l_lst_filt, M, replicate_seed, base_seed, iteration = args
    log_path = f"logs/bootstrap_{base_seed}.log"

    with open(log_path, "a") as f:
        f.write(f"Starting bootstrap iteration {iteration}\n")

    random.seed(replicate_seed)
    n_samples = len(psi_lst_filt)
    indices = random.choices(range(n_samples), k=n_samples)
    psi_lst_boot = [psi_lst_filt[i] for i in indices]
    l_lst_boot = [l_lst_filt[i] for i in indices]

    res_boot_mixture, res_boot_null, res_boot_sum = model.fit_model_M(psi_lst_boot, l_lst_boot, M)

    with open(log_path, "a") as f:
        f.write(f"Finished bootstrap iteration {iteration}\n")

    res_boot_all = pd.concat([res_boot_mixture, res_boot_null, res_boot_sum], ignore_index=True)
    return res_boot_all

if __name__ == "__main__":
    seed = int(sys.argv[1])
    random.seed(seed)

    M = 1500

    # Load data
    tracts_df = read_tracts(seed, MAF=0.5)
    maf_df = read_MAF(seed)

    # Preprocess
    l_lst_filt, psi_lst_filt = get_l_psi(maf_df, tracts_df, M, 5000, 0.5)

    # Fit original model
    res_mixture, res_null, res_sum = model.fit_model_M(psi_lst_filt, l_lst_filt, M)

    res_mixture.to_csv(f"results/res_mixture_seed{seed}.csv")
    res_null.to_csv(f"results/res_null_seed{seed}.csv")
    res_sum.to_csv(f"results/res_sum_seed{seed}.csv")
    l_psi_df = pd.DataFrame(list(zip(l_lst_filt, psi_lst_filt)), columns=["l", "psi"])
    l_psi_df.to_csv(f"results/l_psi_seed{seed}.csv")

    # Prepare arguments for parallel bootstrap
    args_list = [(psi_lst_filt, l_lst_filt, 1500, seed * 10000 + i, seed, i + 1) for i in range(500)]

    # Run bootstrap in parallel with progress logging
    bootstrap_results = []
    with ProcessPoolExecutor() as executor:
        for i, result in enumerate(executor.map(bootstrap_once, args_list)):
            print(f"Finished bootstrap iteration {i + 1}")
            result["bootstrap"] = f"boot_{i + 1}"
            bootstrap_results.append(result)

    bootstrap_df = pd.concat(bootstrap_results, ignore_index=True)

    # Save to CSV
    bootstrap_df.to_csv(f"results/bootstrap_res_seed{seed}.csv", index=False)
    