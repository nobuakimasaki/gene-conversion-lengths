#!/usr/bin/env python3
"""
Simulate gene conversion tracts (many per individual) and record observed lengths.
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
    p.add_argument("--dist",
                   choices=["geom", "geom2", "geom3", "unif", "mixture"],
                   default="geom")
    p.add_argument("--cpus", type=int,
                   default=max(1, mp.cpu_count() - 1),
                   help="Workers for multiprocessing Pool (default: all-1).")
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--t_per_ind", type=int, default=100)
    p.add_argument("--n_ind", type=int, default=10_000)
    p.add_argument("--maf_path", default="sim5_seed1_10Mb_n125000.gtstats")
    p.add_argument("--vcf_path", default="sim5_seed1_10Mb_n125000_err0.0002phased_del1.vcf.gz")
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

# ---------------------- per-individual sim --------------------------
def sim_for_one_ind(ind_idx, tracts_for_ind, positions, genotypes,
                    debug=False, dbg_n=5, ind_label=None):
    g = genotypes[:, ind_idx]
    het = ((g[:, 0] != g[:, 1]) & (g[:, 0] >= 0) & (g[:, 1] >= 0))
    het_idx = np.flatnonzero(het)

    if debug:
        label = ind_label if ind_label is not None else ind_idx
        print(f"\n[IND {label}] n_variants={len(positions)}, n_het={het_idx.size}")

    out = np.zeros((tracts_for_ind.shape[0], 3), dtype=np.int64)
    printed = 0

    for j, (st, en) in enumerate(tracts_for_ind):
        left  = np.searchsorted(positions, st, side='left')
        right = np.searchsorted(positions, en, side='right') - 1

        if right < left:
            if debug and printed < dbg_n:
                print(f"  Tract {j}: st={st}, en={en} ⇒ no variants → [0,0,0]")
                printed += 1
            continue

        i0 = np.searchsorted(het_idx, left,  side='left')
        i1 = np.searchsorted(het_idx, right, side='right') - 1

        if i1 < i0:
            if debug and printed < dbg_n:
                print(f"  Tract {j}: st={st}, en={en} ⇒ no hets → [0,0,0]")
                printed += 1
            continue

        obs_start = positions[het_idx[i0]]
        obs_end   = positions[het_idx[i1]]
        out[j]    = (obs_start, obs_end, obs_end - obs_start + 1)

        if debug and printed < dbg_n:
            print(f"  Tract {j}: st={st}, en={en} | "
                  f"obs_start={obs_start}, obs_end={obs_end}, L={out[j,2]}")
            printed += 1

    return out

# ------------------------------ main --------------------------------
def main(args):
    print("n cores:", mp.cpu_count())
    print(f"Using {args.cpus} worker(s)")
    print(f"\n==== Simulating for distribution: {args.dist} ====\n")

    # MAF filter
    maf_df = pd.read_table(args.maf_path, header=None)
    keep_positions = maf_df.loc[maf_df.iloc[:, 10] >= 0.05, 1].astype(int).to_numpy()

    # VCF
    callset = allel.read_vcf(args.vcf_path)
    genotype_calls    = callset['calldata/GT']
    variant_positions = callset['variants/POS'].astype(int)

    mask = np.isin(variant_positions, keep_positions)
    filtered_positions = variant_positions[mask]
    filtered_genotypes = genotype_calls[mask]

    print("filtered_genotypes shape:", filtered_genotypes.shape)

    T      = args.t_per_ind
    n_ind  = args.n_ind
    N_tot  = T * n_ind
    iters  = args.iters

    worker = partial(sim_for_one_ind,
                     positions=filtered_positions,
                     genotypes=filtered_genotypes,
                     debug=True)

    chunksize = max(1, n_ind // (args.cpus * 4))

    all_rows = []

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

        # stitch
        for ind_idx, arr in zip(inds, job_results):
            valid = arr[:, 2] != 0
            if not np.any(valid):
                continue
            rows = np.column_stack([
                arr[valid],
                np.repeat(iteration, valid.sum()),
                np.repeat(ind_idx,  valid.sum()),
            ])
            all_rows.append(rows)

    if not all_rows:
        print("No observed tracts > 0. Nothing to write.")
        return

    all_rows = np.vstack(all_rows)
    dist_col = np.full(all_rows.shape[0], args.dist, dtype=object)

    out_path = f"sim_tracts_vcf_{args.dist}_multiple_iterations.csv"
    with open(out_path, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(["obs_start", "obs_end", "length",
                         "iteration", "ind_idx", "distribution"])
        for r, d in zip(all_rows, dist_col):
            writer.writerow(list(r) + [d])

    print(f"Finished writing {out_path}")

# ---------------------------- entrypoint -----------------------------
if __name__ == "__main__":
    args = parse_args()
    main(args)
