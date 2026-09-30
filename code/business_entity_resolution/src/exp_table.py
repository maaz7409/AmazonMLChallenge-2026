"""Markdown rows for experiment_log.md from artifacts/experiments/results.jsonl.

Usage:  python -m src.exp_table [tag_prefix ...]
One row per experiment (the latest run of each tag), in file order: validation macro F0.5,
per country, LOCO, singleton accuracy, pair P/R, and the decision-step result if any.
"""

import json
import sys

from .utils import load_config


def row(r: dict) -> str:
    loco = r.get("loco", {})
    dec = r.get("decision")
    notes = [f"{r['n_features']} feats", f"best iter {r['best_iteration']}",
             f"singleton acc {r['singleton_acc']}", f"P {r['pair_precision']} R {r['pair_recall']}"]
    if dec:
        params = {k: v for k, v in dec["params"].items() if k != "method"}
        notes.append(f"decision {dec['params']['method']} {params}: **{dec['macro_f05']}**")
    if r.get("stack"):
        s = r["stack"]
        notes.append(f"stage 2: **{s['macro_f05']}** (US {s['by_country'].get('US')}, India "
                     f"{s['by_country'].get('India')}; singleton acc {s['singleton_acc']}; P {s['pair_precision']} "
                     f"R {s['pair_recall']}; LOCO {r.get('stack_loco', '–')})")
    return (f"| {r['tag']} | {r['macro_f05']} | {r['by_country'].get('US', '–')} | "
            f"{r['by_country'].get('India', '–')} | {loco.get('US->India', '–')} | "
            f"{loco.get('India->US', '–')} | {'; '.join(notes)} |")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cfg = load_config()
    path = cfg["paths"]["artifacts_dir"] / "experiments" / "results.jsonl"
    latest = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        latest[r["tag"]] = r  # a re-run of a tag replaces the earlier result
    prefixes = sys.argv[1:]
    for tag, r in latest.items():
        if not prefixes or any(tag.startswith(p) for p in prefixes):
            print(row(r))


if __name__ == "__main__":
    main()
