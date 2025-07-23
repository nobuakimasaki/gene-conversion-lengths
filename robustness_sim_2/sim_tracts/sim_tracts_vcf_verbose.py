#!/usr/bin/env python3
"""
Simulate gene conversion tracts (many per individual) and record observed lengths.
Verbose version: writes left/right/het/genotype strings per tract.
Also dumps MAF-filter debug files.
"""

import allel
import numpy as np
import pandas as pd
import csv
import multiprocessing as mp
from functools import partial
import argparse

# ----------------------------- CLI ----------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dist", choices=["geom", "geom2", "geom3", "unif", "mixture"], default="geom")
    p.add_argument("--cpus", type=int, default=max(1, mp.cpu_count() - 1),
                   help="Workers for multiprocessing Pool (default: all-1).")
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--t_per_ind", type=int, default=100)
    p.add_argument("--n_ind", type=int, default=10_000)
    p.add_argument("--maf_path", default="../data/sim5_seed1_10Mb_n125000.gtstats")
    p.add_argument("--vcf_path", default="../data/sim5_seed1_10Mb_n125000_err0.0002phased_del1.vcf.gz")
    p.add_argument("--maf_debug", action="store_true", help="Write maf_filter_check.csv etc.")
    p.add_argument("--debug_n", type=int, default=None,
                   help="If set, limit verbose rows per individual to this many (still outputs all tracts).")
    return p.parse_args()

# ----------------------------- RNG ----------------------------------
rng = np.random.default_rng(27)

# ----------------------- tract simulator ----------------------------
def sim_tracts_fast(n, lo, hi, dist_name, mean=21):
    m = n * 2
    starts = rng.integers(lo, hi - 2000, size=m)

    if dist_name == "geom":
        lens = rng.geometric(1 / mean, m)
    elif dist_name == "geom2":
        p = 2 / mean
        lens = rng.geometric(p, m) + rng.geometric(p, m)
    elif dist_name == "geom3":
        p = 3 / mean
        lens = rng.geometric(p, m) + rng.geometric(p, m) + rng.geometric(p, m)
    elif dist_name == "unif":
        lens = rng.integers(1, mean * 2, size=m)
    elif dist_name == "mixture":
        w1, mu1 = 0.005, 725
        mu2 = (mean - w1 * mu1) / (1 - w1)
        p1, p2 = 1 / mu1, 1 / mu2
        choose1 = rng.random(m) < w1
        lens = np.where(choose1, rng.geometric(p1, m), rng.geometric(p2, m))
    else:
        raise ValueError(f"Invalid distribution type: {dist_name}")

    ends = starts + lens - 1
    good = ends < hi
    starts, ends = starts[good][:n], ends[good][:n]
    return np.column_stack((starts, ends)).astype(np.int64)

# ---------------------- per-individual sim (verbose) ----------------
def sim_for_one_ind_verbose(ind_idx, tracts_for_ind, positions, genotypes,
                            debug_n=None):
    """
    Return a list of tuples for one individual:
    (obs_start, obs_end, L, left, right, het_positions_str, geno_str)
    """
    g = genotypes[:, ind_idx]                    # (n_var, 2)
    het = ((g[:, 0] != g[:, 1]) & (g[:, 0] >= 0) & (g[:, 1] >= 0))
    het_idx = np.flatnonzero(het)

    rows = []
    printed = 0
    for j, (st, en) in enumerate(tracts_for_ind):
        left  = np.searchsorted(positions, st, side='left')
        right = np.searchsorted(positions, en, side='right') - 1

        if right < left:   # no variants in interval
            rows.append((np.nan, np.nan, 0,
                         left, right, "", ""))
            if debug_n and printed < debug_n:
                printed += 1
            continue

        i0 = np.searchsorted(het_idx, left,  side='left')
        i1 = np.searchsorted(het_idx, right, side='right') - 1

        if i1 < i0:  # variants but no hets
            geno_slice = g[left:right+1]
            geno_str   = ";".join(f"{a}/{b}" for a, b in geno_slice) if geno_slice.size else ""
            rows.append((np.nan, np.nan, 0,
                         left, right, "", geno_str))
            if debug_n and printed < debug_n:
                printed += 1
            continue

        obs_start = positions[het_idx[i0]]
        obs_end   = positions[het_idx[i1]]
        L         = obs_end - obs_start + 1

        het_pos_slice = positions[het_idx[i0:i1+1]]
        geno_slice    = g[left:right+1]

        het_str  = ";".join(map(str, het_pos_slice)) if het_pos_slice.size else ""
        geno_str = ";".join(f"{a}/{b}" for a, b in geno_slice) if geno_slice.size else ""

        rows.append((obs_start, obs_end, L,
                     left, right, het_str, geno_str))

        if debug_n and printed < debug_n:
            printed += 1

    return rows

# ------------------------------ main --------------------------------
def main(args):
    print("n cores:", mp.cpu_count())
    print(f"Using {args.cpus} worker(s)")
    print(f"\n==== Simulating for distribution: {args.dist} ====\n")

    # --------- MAF filter (fast + debug) ----------
    # Read only needed columns
    maf_df = pd.read_csv(args.maf_path, sep='\t', header=None,
                         usecols=[1, 10], dtype={1: 'int32', 10: 'float32'})
    keep_positions = maf_df.loc[maf_df[10] >= 0.05, 1].to_numpy()

    # VCF read
    callset = allel.read_vcf(args.vcf_path, fields=['variants/POS', 'calldata/GT'])
    genotype_calls    = callset['calldata/GT']
    variant_positions = callset['variants/POS'].astype(int)

    # Faster mask if both sorted (we're checking if variant_positions is in keep_positions)
    # mask contains TRUE for indices where variant_position is in keep_positions
    idx  = np.searchsorted(keep_positions, variant_positions)
    mask = (idx < keep_positions.size) & (keep_positions[idx] == variant_positions)

    filtered_positions = variant_positions[mask]
    filtered_genotypes = genotype_calls[mask]

    print("filtered_genotypes shape:", filtered_genotypes.shape)

    if args.maf_debug:
        df_check = pd.DataFrame({
            'pos': variant_positions,
            'in_keep': np.in1d(variant_positions, keep_positions),
            'masked': mask
        })
        df_check.to_csv("maf_filter_check.csv", index=False)
        np.savetxt("keep_positions.txt", keep_positions, fmt="%d")
        np.savetxt("filtered_positions.txt", filtered_positions, fmt="%d")

    # --------- Simulation settings ----------
    T      = args.t_per_ind
    n_ind  = args.n_ind
    N_tot  = T * n_ind
    iters  = args.iters

    worker = partial(sim_for_one_ind_verbose,
                     positions=filtered_positions,
                     genotypes=filtered_genotypes,
                     debug_n=args.debug_n)

    chunksize = max(1, n_ind // (args.cpus * 4))

    # Output file (streaming)
    out_path = f"sim_tracts_vcf_{args.dist}_verbose.csv"
    header = ["obs_start", "obs_end", "length",
              "left", "right", "het_positions", "geno",
              "iteration", "ind_idx", "distribution"]

    first_write = True

    for iteration in range(iters):
        print(f"Iteration {iteration} / {iters-1}")

        tracts = sim_tracts_fast(N_tot,
                                 filtered_positions.min(),
                                 filtered_positions.max(),
                                 dist_name=args.dist)
        tracts = tracts.reshape(n_ind, T, 2)

        inds = rng.integers(filtered_genotypes.shape[1], size=n_ind)

        with mp.Pool(args.cpus) as pool:
            job_results = pool.starmap(worker,
                                       ((inds[i], tracts[i]) for i in range(n_ind)),
                                       chunksize=chunksize)

        # Write this iteration's rows
        with open(out_path, 'a', newline='') as fh:
            writer = csv.writer(fh)
            if first_write:
                writer.writerow(header)
                first_write = False

            for ind_idx, rows in zip(inds, job_results):
                # rows is list of tuples from verbose worker
                for r in rows:
                    writer.writerow(list(r) + [iteration, ind_idx, args.dist])

    print(f"Finished writing {out_path}")

# ---------------------------- entrypoint -----------------------------
if __name__ == "__main__":
    args = parse_args()
    main(args)
