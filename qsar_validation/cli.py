"""Command-line entry point.

Example:
    python -m qsar_validation examples/example_dataset.csv \
        --activity pIC50 --split split --id compound_id --out results/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import plots
from .core import validate


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="qsar_validation",
        description="Validate an MLR QSAR model: applicability domain (Williams plot), "
                    "Y-randomization and external validation metrics.")
    p.add_argument("csv", help="Table with one row per compound.")
    p.add_argument("--activity", required=True, help="Column with the response (e.g., pIC50).")
    p.add_argument("--split", required=True,
                   help="Column marking each row as 'train' or 'test'.")
    p.add_argument("--descriptors", default=None,
                   help="Comma-separated descriptor columns (default: every other numeric column).")
    p.add_argument("--id", default=None, help="Optional compound ID column, used in the AD report.")
    p.add_argument("--n-yrand", type=int, default=100, help="Y-randomization iterations (default 100).")
    p.add_argument("--seed", type=int, default=0, help="Random seed (default 0).")
    p.add_argument("--out", default="results", help="Output folder (default: results/).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    df = pd.read_csv(args.csv)

    for col in [args.activity, args.split] + ([args.id] if args.id else []):
        if col not in df.columns:
            raise SystemExit(f"Column '{col}' not found. Available: {', '.join(df.columns)}")

    if args.descriptors:
        desc = [c.strip() for c in args.descriptors.split(",")]
    else:
        skip = {args.activity, args.split, args.id}
        desc = [c for c in df.select_dtypes("number").columns if c not in skip]
    if not desc:
        raise SystemExit("No descriptor columns found.")

    split = df[args.split].astype(str).str.lower()
    train, test = df[split == "train"], df[split == "test"]
    if len(train) == 0 or len(test) == 0:
        raise SystemExit("Need at least one 'train' and one 'test' row in the split column.")
    if len(train) <= len(desc) + 1:
        raise SystemExit("Too few training compounds for the number of descriptors.")

    res = validate(train[desc], train[args.activity], test[desc], test[args.activity],
                   n_yrand=args.n_yrand, seed=args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plots.williams_plot(res, out / "williams_plot.png")
    plots.predicted_vs_experimental(res, train[args.activity], test[args.activity],
                                    out / "predicted_vs_experimental.png")
    plots.y_randomization_plot(res, out / "y_randomization.png")

    ids = (lambda d: d[args.id].values) if args.id else (lambda d: d.index.values)
    report = pd.DataFrame({
        "compound": np.concatenate([ids(train), ids(test)]),
        "set": ["train"] * len(train) + ["test"] * len(test),
        "experimental": np.concatenate([train[args.activity], test[args.activity]]),
        "predicted": np.concatenate([res.y_train_pred, res.y_test_pred]),
        "leverage": np.concatenate([res.lev_train, res.lev_test]),
        "std_residual": np.concatenate([res.sres_train, res.sres_test]),
    })
    report["inside_AD"] = (report["leverage"] <= res.h_star) & (report["std_residual"].abs() <= 3)
    report.to_csv(out / "applicability_domain.csv", index=False)

    metrics = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.metrics.items()}
    metrics["h_star"] = round(res.h_star, 4)
    metrics["descriptors"] = desc
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"Descriptors ({len(desc)}): {', '.join(desc)}")
    for k, v in metrics.items():
        if k != "descriptors":
            print(f"  {k:<20} {v}")
    print(f"Results written to {out.resolve()}")
    return 0
