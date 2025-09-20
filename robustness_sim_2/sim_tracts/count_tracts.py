from pathlib import Path
import pandas as pd
import numpy as np

# Grab both patterns (geom*, mixture*)
paths = sorted(Path(".").glob("sim_tracts_vcf_*_verbose.csv"))
if not paths:
    raise FileNotFoundError("No files matching sim_tracts_vcf_*_verbose.csv")

dfs = []
ref_cols = None

for p in paths:
    df = pd.read_csv(p)               # header in row 1
    if ref_cols is None:
        ref_cols = df.columns         # remember canonical column order
    else:
        # Reorder/align by column name to match the first file exactly
        df = df.reindex(columns=ref_cols)
    dfs.append(df)

# Stack all rows
all_df = pd.concat(dfs, ignore_index=True)

# If you want a plain numeric array (drop any non-numeric columns automatically):
numeric_df = all_df.select_dtypes(include="number")

# -> 2-D NumPy array
arr = numeric_df.to_numpy()           # shape: (total_rows, num_numeric_columns)

# (Optional) keep the column names for reference
cols = numeric_df.columns.to_list()

print(arr.shape, cols)

# all_df is your combined DataFrame
all_df["length"] = pd.to_numeric(all_df["length"], errors="coerce")

counts = (
    all_df.groupby(["distribution", "true_mean", "iteration"])
          .agg(n_tracts_2_1500=("length", lambda s: ((s >= 2) & (s <= 1500)).sum()))
          .reset_index()
          .sort_values(["distribution", "true_mean", "iteration"])
)

print(counts)

# Option A: just the minimum value per (distribution, true_mean)
min_counts = (
    counts.groupby(["distribution", "true_mean"], as_index=False)["n_tracts_2_1500"]
          .min()
          .rename(columns={"n_tracts_2_1500": "min_n_tracts_2_1500"})
)
print(min_counts)

