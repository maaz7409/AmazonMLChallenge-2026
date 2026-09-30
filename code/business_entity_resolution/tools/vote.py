"""Majority vote over the match lists of several matching_results.tsv files (one row per S1,
same S1 order): a target is kept for an S1 when at least `--min` of the files emit it.

    python tools/vote.py --min 2 -o out.tsv a.tsv b.tsv c.tsv

The candidate set is unchanged (every emitted pair of every input is a candidate), so the
usual candidate_pairs.tsv still applies. Validate the result with utils/validate_submission.py."""
import argparse
from collections import Counter
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--min", type=int, default=2, help="votes needed to keep a match")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()
    frames = [pd.read_csv(f, sep="\t", dtype=str, keep_default_na=False) for f in args.files]
    ids = frames[0]["source1_entity_id"].to_numpy()
    for f in frames[1:]:
        assert (f["source1_entity_id"].to_numpy() == ids).all(), "S1 order differs between files"
    lists = [f["matched_entity_ids"].to_numpy() for f in frames]
    out = []
    for row in zip(*lists):
        votes = Counter(t for lst in row for t in lst.split(",") if t)
        first = next((lst for lst in row if lst), "")  # keep the first file's order for the kept IDs
        order = [t for t in first.split(",") if t] + [t for t in votes if t not in set(first.split(","))]
        out.append(",".join(t for t in order if votes[t] >= args.min))
    res = pd.DataFrame({"source1_entity_id": ids, "matched_entity_ids": out})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out, sep="\t", index=False, lineterminator="\n")
    n = [int((f["matched_entity_ids"] != "").sum()) for f in frames]
    print(f"inputs: S1 with matches {n}; vote (min {args.min}): {int((res['matched_entity_ids'] != '').sum()):,}; wrote {args.out}")


if __name__ == "__main__":
    main()
