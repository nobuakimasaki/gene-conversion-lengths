import model
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.optimize import dual_annealing
from itertools import chain
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import random
import sys

def read_tracts(chr_num):
    path = f"/projects/browning/brwnlab/sharon/for_nobu/gc_length/ukbiobank/chr{chr_num}.ibdclust2cM_trim1_combinedoffsets_v6.inf_obs_tracts2"
    df = pd.read_table(path, header=None, sep=r'\s+', engine='python')
    return df

def read_MAF(chr_num):
    path = f"/projects/browning/brwnlab/sharon/for_nobu/gc_length/ukbiobank/chr{chr_num}.allregions.pmaf.gz"
    df = pd.read_table(path, compression='gzip', header=None, sep=r'\s+', engine='python')
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
    maf_values = MAF_df[2]
    positions = MAF_df[1]
    psi[positions] = 2 * maf_values * (1 - maf_values)

    # Set psi to 0 where MAF is < 0.05 or > MAF_ceil
    exc = MAF_df[(MAF_df[2] < 0.05) | (MAF_df[2] > MAF_ceil)]
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

def stratify_tracts_by_hotspot(tracts_df, hotspots_chr):
    tracts_df = tracts_df.copy()
    tracts_df.columns = range(tracts_df.shape[1])
    tracts_df['overlaps_hotspot'] = False

    for _, row in hotspots_chr.iterrows():
        h_start = row["first_pos"]
        h_end = row["last_pos"]
        overlaps = (tracts_df[0] <= h_end) & (tracts_df[1] >= h_start)
        tracts_df.loc[overlaps, 'overlaps_hotspot'] = True

    return tracts_df

def bootstrap_once(args):
    psi_lst_filt, l_lst_filt, M, replicate_seed, base_seed, iteration, label = args
    log_path = f"logs/bootstrap_{label}_{base_seed}.log"

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
    res_boot_all["bootstrap"] = f"boot_{iteration}"
    return res_boot_all

if __name__ == "__main__":
    seed = int(sys.argv[1])
    num_cores = multiprocessing.cpu_count()

    MAF_df_lst = [read_MAF(chr) for chr in range(1, 23)]

    # Load hotspot data
    hotspot_df = pd.read_csv("genomewide_hotspots.csv")
    hotspot_df = hotspot_df[hotspot_df["hotspot"] == True]

    # Ensure chr is integer for compatibility
    hotspot_df["chr"] = hotspot_df["chr"].astype(int)
    
    hotspot_dict = {chr_num: df for chr_num, df in hotspot_df.groupby("chr")}

    tracts_df_hotspot = []
    tracts_df_nonhotspot = []

    for chr_num in range(1, 23):
        df_chr = read_tracts(chr_num)
        if chr_num in hotspot_dict:
            df_chr_strat = stratify_tracts_by_hotspot(df_chr, hotspot_dict[chr_num])
            tracts_df_hotspot.append(df_chr_strat[df_chr_strat["overlaps_hotspot"] == True].drop(columns=["overlaps_hotspot"]))
            tracts_df_nonhotspot.append(df_chr_strat[df_chr_strat["overlaps_hotspot"] == False].drop(columns=["overlaps_hotspot"]))
        else:
            tracts_df_hotspot.append(df_chr.iloc[0:0])  # empty
            tracts_df_nonhotspot.append(df_chr)

    # ---- Process and fit model for hotspot tracts ----
    def get_l_psi_hotspot(i):
        l_vals, psi_vals = get_l_psi(
            MAF_df=MAF_df_lst[i],
            df=tracts_df_hotspot[i],
            M=1500,
            region=5000,
            MAF_ceil=0.5
        )
        chr_col = [i + 1] * len(l_vals)  # i is 0-based; chromosomes are 1-based
        return list(zip(chr_col, l_vals, psi_vals))

    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        l_psi_hotspot = list(executor.map(get_l_psi_hotspot, range(len(MAF_df_lst))))

    flat_hotspot = list(chain.from_iterable(l_psi_hotspot))
    chr_hotspot, l_hotspot, psi_hotspot = zip(*flat_hotspot) if flat_hotspot else ([], [], [])

    if seed == 1:
        res_mixture_h, res_null_h, res_sum_h = model.fit_model_M(psi_hotspot, l_hotspot, 1500)
        res_mixture_h.to_csv("res_mixture_hotspot.csv")
        res_null_h.to_csv("res_null_hotspot.csv")
        res_sum_h.to_csv("res_sum_hotspot.csv")
        
        l_psi_hotspot_df = pd.DataFrame({
            "chr": chr_hotspot,
            "l": l_hotspot,
            "psi": psi_hotspot
        })
        l_psi_hotspot_df.to_csv("l_psi_hotspot.csv", index=False)

    # ---- Process and fit model for non-hotspot tracts ----
    def get_l_psi_nonhotspot(i):
        l_vals, psi_vals = get_l_psi(
            MAF_df=MAF_df_lst[i],
            df=tracts_df_nonhotspot[i],
            M=1500,
            region=5000,
            MAF_ceil=0.5
        )
        chr_col = [i + 1] * len(l_vals)
        return list(zip(chr_col, l_vals, psi_vals))

    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        l_psi_nonhotspot = list(executor.map(get_l_psi_nonhotspot, range(len(MAF_df_lst))))

    flat_nonhotspot = list(chain.from_iterable(l_psi_nonhotspot))
    chr_nonhotspot, l_nonhotspot, psi_nonhotspot = zip(*flat_nonhotspot) if flat_nonhotspot else ([], [], [])

    if seed == 1:
        res_mixture_nh, res_null_nh, res_sum_nh = model.fit_model_M(psi_nonhotspot, l_nonhotspot, 1500)
        res_mixture_nh.to_csv("res_mixture_nonhotspot.csv")
        res_null_nh.to_csv("res_null_nonhotspot.csv")
        res_sum_nh.to_csv("res_sum_nonhotspot.csv")

        l_psi_nonhotspot_df = pd.DataFrame({
            "chr": chr_nonhotspot,
            "l": l_nonhotspot,
            "psi": psi_nonhotspot
        })
        l_psi_nonhotspot_df.to_csv("l_psi_nonhotspot.csv", index=False)

    # ========== BOOTSTRAP FOR HOTSPOT TRACTS ==========
    args_hotspot = [(psi_hotspot, l_hotspot, 1500, seed * 10000 + i, seed, i + 1, "hotspot") for i in range(12)]

    bootstrap_results_hotspot = []
    with ProcessPoolExecutor() as executor:
        for i, result in enumerate(executor.map(bootstrap_once, args_hotspot)):
            print(f"Finished hotspot bootstrap iteration {i + 1}")
            bootstrap_results_hotspot.append(result)

    bootstrap_df_hotspot = pd.concat(bootstrap_results_hotspot, ignore_index=True)
    bootstrap_df_hotspot.to_csv(f"bootstrap_res_hotspot_seed{seed}.csv", index=False)

    # ========== BOOTSTRAP FOR NON-HOTSPOT TRACTS ==========
    args_nonhotspot = [(psi_nonhotspot, l_nonhotspot, 1500, seed * 20000 + i, seed, i + 1, "nonhotspot") for i in range(12)]

    bootstrap_results_nonhotspot = []
    with ProcessPoolExecutor() as executor:
        for i, result in enumerate(executor.map(bootstrap_once, args_nonhotspot)):
            print(f"Finished non-hotspot bootstrap iteration {i + 1}")
            bootstrap_results_nonhotspot.append(result)

    bootstrap_df_nonhotspot = pd.concat(bootstrap_results_nonhotspot, ignore_index=True)
    bootstrap_df_nonhotspot.to_csv(f"bootstrap_res_nonhotspot_seed{seed}.csv", index=False)
