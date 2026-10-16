"""Generate a small SYNTHETIC dataset for demonstrating the toolkit.

The values are simulated (not real compounds or measured activities) so the
repository contains no proprietary or unpublished data. To use real data,
compute descriptors for public ChEMBL compounds (e.g., with RDKit) and save
them in the same column layout.
"""

from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 80
desc = {
    "MolLogP": rng.normal(2.5, 1.2, n),
    "TPSA": rng.normal(80, 25, n),
    "NumHDonors": rng.integers(0, 5, n).astype(float),
    "nAromRings": rng.integers(0, 4, n).astype(float),
}
df = pd.DataFrame(desc)
df["pIC50"] = (4.0 + 0.45 * df.MolLogP - 0.012 * df.TPSA + 0.30 * df.nAromRings
               + rng.normal(0, 0.35, n)).round(3)

# Two deliberately unusual compounds, to show what falls outside the domain.
df.loc[n - 2, ["MolLogP", "TPSA"]] = [7.5, 20.0]
df.loc[n - 1, "pIC50"] += 2.0

df.insert(0, "compound_id", [f"CPD-{i + 1:03d}" for i in range(n)])
df["split"] = "train"
df.loc[rng.choice(np.arange(n - 2), size=18, replace=False), "split"] = "test"
df.loc[[n - 2, n - 1], "split"] = "test"

out = Path(__file__).with_name("example_dataset.csv")
df.round(3).to_csv(out, index=False)
print(f"Wrote {len(df)} synthetic compounds to {out}")
