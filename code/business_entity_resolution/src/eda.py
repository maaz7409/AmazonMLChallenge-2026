"""Data checks (`--stage eda`).

Loads every raw file with the prescribed read call, verifies row counts against raw
line counts, and measures the properties later design decisions depend on:

* rows per file and per country; empty-name, empty-address and Indian-script rates;
* singleton share, matches-per-S1 distribution and the S2 vs S3 share;
* exclusivity: whether any S2/S3 record is labeled for more than one S1 record
  (gates the exclusivity post-processing, decide.assign_targets);
* cross-country pairs: whether any labeled pair spans two country labels
  (gates using country as a blocking partition);
* peak RAM while loading, and random matched pairs / singletons to eyeball.

Only compact int arrays are kept between files, so peak memory stays near that of the
largest single file. The report is printed and saved to artifacts/eda/phase0_report.
{md,json}; loading also writes the parquet caches in artifacts/raw/ used by later stages.
"""

import json
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from .io import (GT_COLUMNS, KEY_BASE, SOURCE_COLUMNS, count_data_lines, ground_truth_path,
                 id_keys, key_to_id, positions, read_tsv, source_path, split_id_lists,
                 write_raw_cache)
from .utils import log, peak_rss_gb

# Unicode blocks of the Indian scripts (Devanagari through Sinhala). A normal string, so
# the class holds real characters and works in both Python re and pyarrow's RE2.
INDIC_CLASS = "[ऀ-෿]"


class CountryCodes:
    """Assigns small integer codes to country strings, shared across all files."""

    def __init__(self):
        self.names: list[str] = []
        self._index: dict[str, int] = {}

    def encode(self, values: pd.Series) -> np.ndarray:
        """Return the int16 code of each value, registering unseen strings on the fly."""
        codes, uniques = pd.factorize(values)
        lut = np.empty(len(uniques), dtype=np.int16)
        for i, name in enumerate(uniques):
            if name not in self._index:
                self._index[name] = len(self.names)
                self.names.append(name)
            lut[i] = self._index[name]
        return lut[codes]


def load_checked(path, expected_columns) -> tuple[pd.DataFrame, dict]:
    """Read a raw file with the prescribed call and check its rows against raw line counts."""
    n_lines = count_data_lines(path)
    t0 = time.perf_counter()
    df = read_tsv(path)
    if list(df.columns) != expected_columns:
        raise ValueError(f"{path.name}: unexpected columns {list(df.columns)}")
    info = {"rows": len(df), "raw_data_lines": n_lines, "rows_match_lines": len(df) == n_lines,
            "load_seconds": round(time.perf_counter() - t0, 1),
            "peak_rss_gb_after_load": round(peak_rss_gb(), 2)}
    log(f"loaded {path.name}: {len(df):,} rows, {n_lines:,} raw data lines")
    return df, info


def field_stats(df: pd.DataFrame) -> dict:
    """Per-country row counts plus empty-field and Indian-script rates (in %) of a source file."""
    name = df["business_name"].str.strip()
    addr = df["business_address"].str.strip()
    flags = pd.DataFrame({
        "empty_name_pct": name.eq(""),
        "empty_address_pct": addr.eq(""),
        "indic_name_pct": name.str.contains(INDIC_CLASS, regex=True),
        "indic_address_pct": addr.str.contains(INDIC_CLASS, regex=True),
    })
    groups = flags.groupby(df["country"], sort=True)
    per_country = groups.mean().mul(100).round(2)
    per_country.insert(0, "rows", groups.size())
    return {"per_country": per_country.to_dict(orient="index"),
            "overall": flags.mean().mul(100).round(2).to_dict(),
            "duplicate_ids": int(df["entity_id"].duplicated().sum())}


def pct(x) -> float:
    """Mean of a boolean array (or a fraction) as a plain-float percentage."""
    return round(100 * float(np.mean(x)), 2)


def match_histogram(counts: np.ndarray, top: int = 10) -> list[dict]:
    """Histogram of matches per S1: one row per count 0..top plus a final 'top+1 or more'."""
    hist = np.bincount(np.minimum(counts, top + 1), minlength=top + 2)
    return [{"matches": str(k) if k <= top else f"{top + 1}+", "s1_records": int(h),
             "pct": round(100 * h / len(counts), 2)} for k, h in enumerate(hist)]


def run(cfg: dict) -> dict:
    """Run every data check, print the report and save it under artifacts/eda/."""
    done = cfg["paths"]["artifacts_dir"] / "eda" / "phase0_report.json"
    if done.exists():
        log(f"eda already done ({done}); skipped")
        return json.loads(done.read_text(encoding="utf-8"))
    t_start = time.perf_counter()
    rng = np.random.default_rng(cfg["seed"])
    countries = CountryCodes()
    files, checks = {}, {}

    # 1. Ground truth: one row per train S1 record with its comma-separated matches.
    gt, files["train_ground_truth"] = load_checked(ground_truth_path(cfg), GT_COLUMNS)
    gt_s1 = id_keys(gt["source1_entity_id"])
    pair_row, pair_key, stray = split_id_lists(gt["matched_entity_ids"])
    write_raw_cache(cfg, "train_ground_truth", gt)
    del gt
    n_gt = len(gt_s1)
    pair_row = pair_row.astype(np.int64)
    checks["gt_duplicate_s1_rows"] = n_gt - len(np.unique(gt_s1))
    checks["gt_non_s1_ids_in_s1_column"] = int((gt_s1 // KEY_BASE != 1).sum())
    checks["gt_s1_ids_inside_match_lists"] = int((pair_key // KEY_BASE == 1).sum())
    checks["gt_stray_empty_tokens"] = int(stray)
    # Drop an ID listed twice for the same S1 so it cannot inflate any count below.
    _, first = np.unique(pair_row * (4 * KEY_BASE) + pair_key, return_index=True)
    checks["gt_repeated_ids_within_one_list"] = len(pair_key) - len(first)
    first.sort()
    pair_row, pair_key = pair_row[first], pair_key[first]
    counts = np.bincount(pair_row, minlength=n_gt)

    # Pick the examples now so their records can be grabbed while each file is loaded.
    ecfg = cfg["eda"]
    ex_pairs = rng.choice(len(pair_key), size=min(ecfg["n_example_pairs"], len(pair_key)),
                          replace=False)
    singles = np.flatnonzero(counts == 0)
    ex_singles = rng.choice(singles, size=min(ecfg["n_example_singletons"], len(singles)),
                            replace=False)
    wanted = np.array(sorted({*gt_s1[pair_row[ex_pairs]], *pair_key[ex_pairs], *gt_s1[ex_singles]}))
    records = {}  # key -> (name, address, country) for the example records

    # 2. Train sources: field stats, ID/country arrays for the joins, parquet caches.
    keys, country = {}, {}
    for n in (1, 2, 3):
        name = f"train_source{n}"
        df, files[name] = load_checked(source_path(cfg, "train", n), SOURCE_COLUMNS)
        files[name].update(field_stats(df))
        keys[n] = id_keys(df["entity_id"])
        country[n] = countries.encode(df["country"])
        checks[f"{name}_ids_with_wrong_prefix"] = int((keys[n] // KEY_BASE != n).sum())
        for i in np.flatnonzero(np.isin(keys[n], wanted)):
            records[int(keys[n][i])] = (df["business_name"].iat[i], df["business_address"].iat[i],
                                        df["country"].iat[i])
        write_raw_cache(cfg, name, df)
        del df

    # 3. Ground-truth statistics, joined to source records through the int keys.
    s1_pos = positions(keys[1], gt_s1)
    checks["gt_s1_ids_missing_from_source1"] = int((s1_pos < 0).sum())
    checks["source1_ids_missing_from_gt"] = len(keys[1]) - len(np.unique(s1_pos[s1_pos >= 0]))
    tgt_keys = np.concatenate([keys[2], keys[3]])
    tgt_country = np.concatenate([country[2], country[3]])
    tgt_pos = positions(tgt_keys, pair_key)
    checks["gt_target_ids_missing_from_sources"] = int((tgt_pos < 0).sum())

    s1_country = np.where(s1_pos >= 0, country[1][np.maximum(s1_pos, 0)], -1)
    is_s2 = pair_key // KEY_BASE == 2
    n_s2 = np.bincount(pair_row, weights=is_s2, minlength=n_gt).astype(np.int64)
    n_s3 = counts - n_s2
    matched = counts > 0
    gt_stats = {
        "s1_records": n_gt,
        "labeled_pairs": len(pair_key),
        "singletons": int((~matched).sum()),
        "singleton_pct": pct(~matched),
        "matches_per_s1": {"mean": round(float(counts.mean()), 3),
                           "mean_if_any": round(float(counts[matched].mean()), 3),
                           "median": float(np.median(counts)),
                           "p90": float(np.percentile(counts, 90)),
                           "p99": float(np.percentile(counts, 99)), "max": int(counts.max())},
        "histogram": match_histogram(counts),
        "pair_share_pct": {"S2": pct(is_s2), "S3": pct(~is_s2)},
        "matched_s1_pct": {"with_S2": pct(n_s2[matched] > 0),
                           "with_S3": pct(n_s3[matched] > 0),
                           "S2_only": pct(((n_s2 > 0) & (n_s3 == 0))[matched]),
                           "S3_only": pct(((n_s3 > 0) & (n_s2 == 0))[matched]),
                           "both": pct(((n_s2 > 0) & (n_s3 > 0))[matched])},
        "max_S2_matches_for_one_s1": int(n_s2.max()),
        "max_S3_matches_for_one_s1": int(n_s3.max()),
        "per_country": [],
    }
    for code in np.unique(s1_country):
        m = s1_country == code
        c = counts[m]
        gt_stats["per_country"].append({
            "country": countries.names[code] if code >= 0 else "(not in source1)",
            "s1_records": int(m.sum()), "singleton_pct": pct(c == 0),
            "mean_matches": round(float(c.mean()), 3),
            "mean_if_any": round(float(c[c > 0].mean()), 3) if (c > 0).any() else None,
            "mean_S2": round(float(n_s2[m].mean()), 3), "mean_S3": round(float(n_s3[m].mean()), 3),
            "max": int(c.max())})

    # Exclusivity: can one S2/S3 record belong to more than one S1 record?
    uniq, per_target = np.unique(pair_key, return_counts=True)
    multi = uniq[per_target > 1]
    owners = defaultdict(list)
    for i in np.flatnonzero(np.isin(pair_key, multi[:5])):
        owners[key_to_id(int(pair_key[i]))].append(key_to_id(int(gt_s1[pair_row[i]])))
    exclusivity = {"labeled_targets": len(uniq), "targets_in_multiple_s1_lists": len(multi),
                   "max_s1_lists_per_target": int(per_target.max()), "examples": dict(owners)}

    # Cross-country: does any labeled pair join two different country strings?
    pair_s1_c = s1_country[pair_row]
    pair_t_c = np.where(tgt_pos >= 0, tgt_country[np.maximum(tgt_pos, 0)], -1)
    valid = (pair_s1_c >= 0) & (pair_t_c >= 0)
    cross = np.flatnonzero(valid & (pair_s1_c != pair_t_c))
    combos = Counter(zip(pair_s1_c[cross].tolist(), pair_t_c[cross].tolist()))
    cross_country = {
        "pairs_checked": int(valid.sum()), "cross_country_pairs": len(cross),
        "by_country_pair": {f"{countries.names[a]} -> {countries.names[b]}": v
                            for (a, b), v in combos.most_common(10)},
        "examples": [(key_to_id(int(gt_s1[pair_row[i]])), key_to_id(int(pair_key[i])))
                     for i in cross[:5]]}

    # Unlabeled targets: S2/S3 records that match no S1 record (pure distractors).
    labeled = np.zeros(len(tgt_keys), dtype=bool)
    labeled[tgt_pos[tgt_pos >= 0]] = True
    unlabeled = []
    n2 = len(keys[2])
    for src, part in ((2, slice(0, n2)), (3, slice(n2, None))):
        lab, tc = labeled[part], tgt_country[part]
        for code in np.unique(tc):
            m = tc == code
            unlabeled.append({"source": f"S{src}", "country": countries.names[code],
                              "records": int(m.sum()),
                              "unlabeled_pct": pct(~lab[m])})

    # 4. Test sources: field stats, ID reuse versus train, parquet caches.
    for n in (1, 2, 3):
        name = f"test_source{n}"
        df, files[name] = load_checked(source_path(cfg, "test", n), SOURCE_COLUMNS)
        files[name].update(field_stats(df))
        test_keys = id_keys(df["entity_id"])
        checks[f"{name}_ids_with_wrong_prefix"] = int((test_keys // KEY_BASE != n).sum())
        checks[f"{name}_ids_also_in_train_source{n}"] = int(np.isin(test_keys, keys[n]).sum())
        write_raw_cache(cfg, name, df)
        del df, test_keys

    # 5. Examples.
    def show(key):
        name, address, ctry = records.get(int(key), ("?", "?", "?"))
        return {"id": key_to_id(int(key)), "country": ctry, "name": name, "address": address}

    examples = {"pairs": [{"s1": show(gt_s1[pair_row[i]]), "target": show(pair_key[i])}
                          for i in ex_pairs],
                "singletons": [show(gt_s1[i]) for i in ex_singles]}

    report = {"files": files, "checks": checks, "ground_truth": gt_stats,
              "exclusivity": exclusivity, "cross_country": cross_country,
              "unlabeled_targets": unlabeled, "examples": examples,
              "peak_rss_gb": round(peak_rss_gb(), 2),
              "runtime_seconds": round(time.perf_counter() - t_start, 1)}
    out_dir = cfg["paths"]["artifacts_dir"] / "eda"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "phase0_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False,
                  default=lambda o: o.item() if hasattr(o, "item") else str(o))
    text = to_markdown(report)
    (out_dir / "phase0_report.md").write_text(text, encoding="utf-8")
    print(text)
    log(f"report saved to {out_dir}")
    return report


def md_table(rows: list[dict]) -> str:
    """Render a list of flat dicts as a markdown table (ints get thousands separators)."""
    if not rows:
        return "(none)\n"
    cols = list(rows[0])
    fmt = lambda v: f"{v:,}" if isinstance(v, (int, np.integer)) and not isinstance(v, bool) else str(v)
    lines = ["| " + " | ".join(cols) + " |", "|" + " --- |" * len(cols)]
    lines += ["| " + " | ".join(fmt(r.get(c, "")) for c in cols) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def to_markdown(r: dict) -> str:
    """Human-readable data-check report."""
    out = ["# Phase 0 data checks\n", "## Files\n"]
    out.append(md_table([{"file": k, "rows": v["rows"], "raw data lines": v["raw_data_lines"],
                          "match": v["rows_match_lines"], "load s": v["load_seconds"],
                          "peak RSS GB (so far)": v["peak_rss_gb_after_load"]}
                         for k, v in r["files"].items()]))
    out.append("\n## Rows per country and field quality (% of rows)\n")
    out.append(md_table([{"file": k, "country": c, **stats}
                         for k, v in r["files"].items() if "per_country" in v
                         for c, stats in v["per_country"].items()]))
    out.append("\n## Integrity checks (all should be 0)\n")
    out += [f"- {k}: {v:,}" for k, v in r["checks"].items()]
    g = r["ground_truth"]
    out.append(f"\n\n## Ground truth\n\n- S1 records: {g['s1_records']:,}; labeled pairs: "
               f"{g['labeled_pairs']:,}\n- singletons: {g['singletons']:,} ({g['singleton_pct']}%)\n"
               f"- matches per S1: {g['matches_per_s1']}\n- pair share: {g['pair_share_pct']}\n"
               f"- matched S1 records: {g['matched_s1_pct']}\n- max S2 matches for one S1: "
               f"{g['max_S2_matches_for_one_s1']}; max S3: {g['max_S3_matches_for_one_s1']}\n\n")
    out.append(md_table(g["histogram"]))
    out.append("\nPer S1 country:\n\n" + md_table(g["per_country"]))
    e = r["exclusivity"]
    out.append(f"\n## Exclusivity\n\n- labeled targets: {e['labeled_targets']:,}\n"
               f"- targets in more than one S1 list: {e['targets_in_multiple_s1_lists']:,} "
               f"(max lists per target: {e['max_s1_lists_per_target']})\n"
               f"- examples: {e['examples']}\n")
    c = r["cross_country"]
    out.append(f"\n## Cross-country pairs\n\n- labeled pairs checked: {c['pairs_checked']:,}\n"
               f"- pairs whose S1 and target country differ: {c['cross_country_pairs']:,}\n"
               f"- by country pair: {c['by_country_pair']}\n- examples: {c['examples']}\n")
    out.append("\n## Unlabeled targets (S2/S3 records matching no S1)\n\n" + md_table(r["unlabeled_targets"]))
    out.append(f"\n## Resources\n\n- peak RSS: {r['peak_rss_gb']} GB\n- runtime: {r['runtime_seconds']} s\n")
    out.append("\n## 20 random labeled pairs\n")
    for i, p in enumerate(r["examples"]["pairs"], 1):
        s, t = p["s1"], p["target"]
        out.append(f"{i:2d}. {s['id']} [{s['country']}] {s['name']} | {s['address']}\n"
                   f"    {t['id']} [{t['country']}] {t['name']} | {t['address']}")
    out.append("\n## 10 random singletons\n")
    for i, s in enumerate(r["examples"]["singletons"], 1):
        out.append(f"{i:2d}. {s['id']} [{s['country']}] {s['name']} | {s['address']}")
    return "\n".join(out) + "\n"
