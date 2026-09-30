"""Optional submission variant: match the "no match" rate of countries absent from training.

    python tools/variant_unseen_rate.py [--tag _pl]      # run from code/business_entity_resolution/

The training countries have the same singleton share (US 5.58 %, India 5.59 %), and the tuned
rule leaves a known share of validation S1 empty. A country absent from training (France)
whose test S1 are left empty noticeably less often is probably merging singletons with
namesakes. For each such country only, this raises its emit threshold t_emit (never lowers
it) until its empty share equals the validation share; t_add and rel are unchanged, and
countries seen in training keep the tuned rule exactly. No labels are used: only the tuned
rule, validation predictions and the test predictions of the run (`--tag _pl` for the
pseudo-label round).

Writes <output_dir>/variants/unseen_rate<tag>/matching_results.tsv (+ the run's candidate
file) and validates it. It is a leaderboard probe: upload it next to the run's own file.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.decide import best_of_group, decide, emit_mask, emission_profile, groups  # noqa: E402
from src.io import load_source, write_id_lists  # noqa: E402
from src.normalize.normalize import load_normalized  # noqa: E402
from src.train import lgbm_dir, preds_dir  # noqa: E402
from src.utils import REPO_ROOT, load_config  # noqa: E402


def main() -> None:
    """Build, report and validate the variant."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="", help='"" for the main run, "_pl" for the pseudo-label round')
    args = ap.parse_args()
    cfg = load_config()
    tag = args.tag
    th = json.loads((lgbm_dir(cfg) / f"thresholds{tag}.json").read_text())["thresholds"]
    if th.get("method", "rule") != "rule":
        raise SystemExit("only the rule decision is supported")

    def load(path):
        df = pd.read_parquet(path, columns=["s1", "t", "p"])
        return df.iloc[np.lexsort((df["t"].to_numpy(), df["s1"].to_numpy()))].reset_index(drop=True)

    val = load(preds_dir(cfg) / (f"val_stack{tag}.parquet" if th.get("stack") else "val.parquet"))
    train_c = load_normalized(cfg, "train", "s1", columns=["country"])["country"].to_numpy(dtype=object)
    vprof = emission_profile(val, emit_mask(val, th), train_c)
    n = sum(v["s1"] for v in vprof.values())
    target = sum(v["empty_share"] * v["s1"] for v in vprof.values()) / n
    test = load(preds_dir(cfg) / f"test{tag}.parquet")
    test_c = load_normalized(cfg, "test", "s1", columns=["country"])["country"].to_numpy(dtype=object)
    s1, p = test["s1"].to_numpy(), test["p"].to_numpy()
    starts, gid = groups(s1)
    best, is_best = best_of_group(p, starts, gid)
    g_c = test_c[s1[starts]]
    t_emit = np.full(len(starts), th["t_emit"])
    changes = {}
    for c in pd.unique(g_c):
        if c in vprof:  # seen in training: keep the tuned rule
            continue
        m = g_c == c
        now = float((best[m] < th["t_emit"]).mean())
        if now >= target:
            changes[str(c)] = {"empty_share": round(now, 4), "target": round(target, 4), "t_emit": th["t_emit"]}
            continue
        t_c = max(th["t_emit"], float(np.quantile(best[m], target)))
        t_emit[m] = t_c
        changes[str(c)] = {"empty_share_before": round(now, 4), "target": round(target, 4),
                           "t_emit": round(t_c, 4), "empty_share_after": round(float((best[m] < t_c).mean()), 4)}
    mask = decide(p, best[gid], is_best, (best >= t_emit)[gid], th["t_add"], th["rel"])
    print("validation profile:", json.dumps(vprof))
    print("unseen-country changes:", json.dumps(changes))
    print("test profile after:", json.dumps(emission_profile(test, mask, test_c)))

    s1_ids = load_source(cfg, "test", 1, columns=["entity_id"])["entity_id"].to_numpy(dtype=object)
    t_ids = np.concatenate([load_source(cfg, "test", k, columns=["entity_id"])["entity_id"]
                            .to_numpy(dtype=object) for k in (2, 3)])
    sm, tm = s1[mask], test["t"].to_numpy()[mask]
    mb = np.searchsorted(sm, np.arange(len(s1_ids) + 1))
    lists = [t_ids[tm[mb[i]:mb[i + 1]]].tolist() for i in range(len(s1_ids))]
    run_out = cfg["paths"]["output_dir"] / (tag.strip("_").replace("pl", "pseudo_label") if tag else "")
    out = cfg["paths"]["output_dir"] / "variants" / f"unseen_rate{tag}"
    write_id_lists(out / "matching_results.tsv", ["source1_entity_id", "matched_entity_ids"], s1_ids.tolist(), lists)
    if (run_out / "candidate_pairs.tsv").exists():
        shutil.copy2(run_out / "candidate_pairs.tsv", out / "candidate_pairs.tsv")
    (out / "variant.json").write_text(json.dumps({"thresholds": th, "target_empty_share": target,
                                                  "changes": changes}, indent=1))
    res = subprocess.run([sys.executable, str(REPO_ROOT.parents[1] / "utils" / "validate_submission.py"),
                          "--matching", str(out / "matching_results.tsv"),
                          "--candidate", str(out / "candidate_pairs.tsv"),
                          "--test-dir", str(cfg["paths"]["data_dir"] / "test")], capture_output=True, text=True)
    print(res.stdout[-1500:], res.stderr[-1000:])
    print(f"wrote {out / 'matching_results.tsv'} (validator exit code {res.returncode})")


if __name__ == "__main__":
    main()
