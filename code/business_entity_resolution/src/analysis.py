"""Validation error analysis: where the tuned pipeline merges wrongly or misses matches.

Run after the train stage:  python -m src.analysis
Prints per-country metrics, error counts by kind, and sampled examples of false merges
and missed matches (with both records' raw names/addresses) for the methodology write-up.
"""

import json
import sys

import numpy as np
import pandas as pd

from .decide import apply_rule, thresholds_path
from .io import load_source, s1_roles, truth_pairs
from .metrics import f05_one
from .normalize.normalize import load_normalized
from .train import preds_dir
from .utils import load_config


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cfg = load_config()
    s1 = load_normalized(cfg, "train", "s1", columns=["key", "country"])
    tg = load_normalized(cfg, "train", "targets", columns=["key"])
    roles = s1_roles(cfg, s1["key"].to_numpy())
    val_rows = np.flatnonzero(roles == 2)
    ts, tt = truth_pairs(cfg, s1["key"].to_numpy(), tg["key"].to_numpy())
    with open(thresholds_path(cfg), encoding="utf-8") as f:
        th = json.load(f)["thresholds"]
    val = pd.read_parquet(preds_dir(cfg) / "val.parquet")
    val["emit"] = apply_rule(val, th)

    in_val = np.isin(ts, val_rows)
    truth = {int(r): set() for r in val_rows}
    for a, b in zip(ts[in_val], tt[in_val]):
        truth[int(a)].add(int(b))
    pred = {}
    for a, b in zip(val["s1"][val["emit"]], val["t"][val["emit"]]):
        pred.setdefault(int(a), set()).add(int(b))
    country = s1["country"].to_numpy()
    f = np.array([f05_one(truth[int(r)], pred.get(int(r), set())) for r in val_rows])
    print("macro F0.5 by country:", {c: round(float(f[country[val_rows] == c].mean()), 5)
                                     for c in pd.unique(country[val_rows])})
    singles = np.array([not truth[int(r)] for r in val_rows])
    emitted = np.array([bool(pred.get(int(r))) for r in val_rows])
    print(f"singletons: {singles.sum():,}, wrongly given a match: {(singles & emitted).sum():,}; "
          f"S1 with matches but nothing emitted: {(~singles & ~emitted).sum():,}")

    fp = val[val["emit"] & (val["label"] == 0)]
    missed_scored = val[~val["emit"] & (val["label"] == 1)]
    n_true_total = int(in_val.sum())
    print(f"false merges: {len(fp):,}; true pairs scored but not emitted: {len(missed_scored):,}; "
          f"true pairs not in candidates: {n_true_total - int(val['label'].sum()):,} of {n_true_total:,}")

    raw1 = load_source(cfg, "train", 1, columns=["business_name", "business_address"])
    raw_t = pd.concat([load_source(cfg, "train", n, columns=["entity_id", "business_name", "business_address"])
                       for n in (2, 3)], ignore_index=True)
    rng = np.random.default_rng(0)
    for title, df in (("FALSE MERGES", fp), ("MISSED (scored, below threshold)", missed_scored)):
        print(f"\n=== {title}: 12 random examples ===")
        for i in rng.choice(len(df), size=min(12, len(df)), replace=False):
            r = df.iloc[i]
            a, b = raw1.iloc[int(r.s1)], raw_t.iloc[int(r.t)]
            print(f"p={r.p:.3f} [{country[int(r.s1)]}]\n  S1: {a.business_name} | {a.business_address}\n"
                  f"  {b.entity_id}: {b.business_name} | {b.business_address}")


if __name__ == "__main__":
    main()
