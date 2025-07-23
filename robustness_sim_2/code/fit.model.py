import sys
import os
import glob
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.optimize import dual_annealing
from itertools import chain
import multiprocessing as mp
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
import random
import argparse

parser = argparse.ArgumentParser(
    description="Fit models + bootstrap for a single distribution."
)
parser.add_argument(
    "--dist",
    required=True,
    help="Name of the distribution to process (e.g. 'geom', 'geom2', ...)",
)
args = parser.parse_args()
DIST_TARGET = args.dist

#sys.path.append(os.path.abspath("../UK_biobank"))
import model

# Function to read MAF file from one region
def read_MAF(seed):
    path = f"/projects/browning/brwnlab/sharon/for_nobu/gc_length/sim5_data/sim5_seed{seed}_10Mb_n125000.gtstats"
    #path = f"data/sim5_seed{seed}_10Mb_n125000.gtstats"
    df = pd.read_table(path, header=None, index_col=False)
    return df

def est_psi(observed, psi, region, length_chrom):
    leftmost = max(int(observed.iat[0]) - region, 0)
    rightmost = min(int(observed.iat[1]) + region, length_chrom - 1)
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

    print("num tracts:")
    print(len(df))

    # Generate psi and length lists
    tracts_lst = [row for _, row in df.iterrows()]
    psi_lst = [est_psi(row, psi=psi, region=region, length_chrom=len(psi))
               for row in tracts_lst]

    print("psi: ")
    print(psi_lst[:10])

    l_lst = [(row.iat[1] - row.iat[0] + 1) for row in tracts_lst]

    print("l: ")
    print(l_lst[:10])

    print("finished processing data for 1 chromosome")

    return l_lst, psi_lst

def bootstrap_once(args):
    """
    One param-bootstrap replicate.

    Returns a tidy DF with a 'model' column already filled.
    """
    psi_lst_filt, l_lst_filt, M, replicate_seed, iteration = args

    random.seed(replicate_seed)
    n_samples     = len(psi_lst_filt)
    idx           = random.choices(range(n_samples), k=n_samples)
    psi_lst_boot  = [psi_lst_filt[i] for i in idx]
    l_lst_boot    = [l_lst_filt[i]   for i in idx]

    res_mix, res_null, res_sum = model.fit_model_M(psi_lst_boot, l_lst_boot, M)
    res_mix["model"], res_null["model"], res_sum["model"] = "mixture", "null", "sum"

    return pd.concat([res_mix, res_null, res_sum], ignore_index=True)

if __name__ == "__main__":

    # ------------------------------------------------------------------
    # 1.  Load all files
    # ------------------------------------------------------------------
    files   = glob.glob("sim_tracts_vcf_*_multiple_iterations.csv")
    df_all  = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)

    df_all  = df_all[df_all["distribution"] == DIST_TARGET]
    if df_all.empty:
        raise ValueError(f"No rows found for distribution '{DIST_TARGET}'.")

    # ------------------------------------------------------------------
    # 2.  Filter out length == 1   OR   length > 1 500
    # ------------------------------------------------------------------
    df_filt = df_all[(df_all["length"] != 1) & (df_all["length"] <= 1_500)]

    # ------------------------------------------------------------------
    # Down-sample every group to exactly 200 rows (no replacement needed)
    # ------------------------------------------------------------------
    TARGET  = 200      # rows to keep per (distribution, iteration) group
    RSTATE  = 27       # random-state seed for reproducibility
    df_down = (
        df_filt
        .groupby(["distribution", "iteration"], group_keys=False)
        .sample(n=TARGET, random_state=RSTATE)    
        .reset_index(drop=True)
    )

    print(df_down.shape)      # (#groups × 200, original_n_columns)

    M        = 1500
    maf_df   = read_MAF(1)
    N_BOOT   = 500
    N_CORES  = max(1, cpu_count() - 1)

    POINT_RESULTS, BOOT_RESULTS = [], []

    for (dist, it), g in df_down.groupby(["distribution", "iteration"]):
        print(f"▶  Processing dist='{dist}', iter={it}")   # progress ping

        # -------- point estimates --------
        l_lst, psi_lst = get_l_psi(maf_df, g, M, 5000, 0.5)
        res_mix, res_null, res_sum = model.fit_model_M(psi_lst, l_lst, M)

        for df, mdl in zip([res_mix, res_null, res_sum],
                           ["mixture", "null", "sum"]):
            out = df.copy()
            out["distribution"], out["iteration"], out["model"] = dist, it, mdl
            POINT_RESULTS.append(out)

        # -------- bootstraps (parallel) --------
        base_seed = hash((dist, it)) & 0xFFFFFFFF

        tasks = [
            (psi_lst, l_lst, M, base_seed + b, b)
            for b in range(N_BOOT)
        ]

        with mp.Pool(N_CORES) as pool:
            for b, boot_df in enumerate(pool.map(bootstrap_once, tasks)):
                boot_df["distribution"], boot_df["iteration"], boot_df["bootstrap_iter"] = dist, it, b
                BOOT_RESULTS.append(boot_df)

    # -------- combine & save --------
    results_all = pd.concat(POINT_RESULTS, ignore_index=True)
    boot_all    = pd.concat(BOOT_RESULTS,  ignore_index=True)

    print("Point-estimate DF :", results_all.shape)
    print("Bootstrap DF      :", boot_all.shape)

    results_all.to_csv(f"fit_model_{DIST_TARGET}_point.csv",     index=False)
    boot_all   .to_csv(f"fit_model_{DIST_TARGET}_bootstrap.csv", index=False)
