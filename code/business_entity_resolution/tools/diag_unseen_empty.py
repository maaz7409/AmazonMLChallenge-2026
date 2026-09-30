"""Leaderboard diagnostic: a copy of a submission with every S1 of a country absent from
training (France) emptied, so one upload measures that country's F0.5 directly.

    python tools/diag_unseen_empty.py [--matching ../../output/matching_results.tsv]

Writes <matching dir>/diag/matching_results_unseen_empty.tsv (validator-clean: rows kept,
lists emptied). With LB_full the leaderboard score of the original file and LB_diag that of
this one, and w the unseen country's share of test S1 (printed below; the public subset is
assumed to have the same share):

    F_unseen = (LB_full - LB_diag) / w + s        s = its singleton share (train: ~0.056)

because an emptied S1 scores 1 if it is a singleton and 0 otherwise. Costs one upload; it
tells whether a change moved France or only US/India. Nothing here is used for training.
"""

import argparse
import csv
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Write the diagnostic file and print the country shares."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--matching", default=str(ROOT.parents[1] / "output" / "matching_results.tsv"))
    ap.add_argument("--data-dir", default=str(ROOT.parents[1] / "dataset"))
    args = ap.parse_args()
    read = dict(sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
    data = Path(args.data_dir)
    train_c = set(pd.read_csv(data / "train" / "train_source1.tsv", usecols=["country"], **read)["country"])
    test = pd.read_csv(data / "test" / "test_source1.tsv", usecols=["entity_id", "country"], **read)
    unseen = set(test["country"]) - train_c
    sub = pd.read_csv(args.matching, **read)
    country = sub["source1_entity_id"].map(dict(zip(test["entity_id"], test["country"])))
    empty = country.isin(unseen)
    sub.loc[empty, "matched_entity_ids"] = ""
    out = Path(args.matching).parent / "diag" / "matching_results_unseen_empty.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        f.writelines(f"{a}\t{b}\n" for a, b in zip(sub["source1_entity_id"], sub["matched_entity_ids"]))
    w = float(empty.mean())
    print(f"unseen countries: {sorted(unseen)}; their share of test S1 w = {w:.4f}; rows emptied: {int(empty.sum()):,}")
    print(f"wrote {out}")
    print(f"after uploading both: F_unseen = (LB_full - LB_diag) / {w:.4f} + 0.056")
    print(f"                      F_seen   = (LB_diag - {w:.4f} * 0.056) / {1 - w:.4f}")


if __name__ == "__main__":
    main()
