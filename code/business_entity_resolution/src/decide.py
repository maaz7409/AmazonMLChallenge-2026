"""Decision rule, threshold search and submission writing.

Rule, per S1 record, over its candidates scored with the LightGBM probability p:
  emit the best candidate only if p_best >= t_emit;
  then add every other candidate with p >= t_add and p >= rel * p_best.
One global threshold set (never per country) is chosen by maximizing validation macro
F0.5, computed with the exact challenge formula; the vectorized grid evaluation is
cross-checked against metrics.macro_f05 for the chosen thresholds.
"""

import itertools
import json
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from .io import load_source, s1_roles, truth_pairs, write_id_lists
from .metrics import macro_f05, pair_precision_recall, singleton_accuracy
from .normalize.normalize import load_normalized
from .train import lgbm_dir, preds_dir
from .utils import log


def groups(s1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(start offset of each S1 group, group id of each row) for rows sorted by S1."""
    starts = np.flatnonzero(np.r_[True, s1[1:] != s1[:-1]])
    gid = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(s1)]))
    return starts, gid


def best_of_group(p: np.ndarray, starts: np.ndarray, gid: np.ndarray):
    """Per-row group maximum and a mask marking one best row per group."""
    best = np.maximum.reduceat(p, starts)
    order = np.lexsort((-p, gid))
    is_best = np.zeros(len(p), dtype=bool)
    is_best[order[starts]] = True  # first row of each group in (group, -p) order
    return best, is_best


def decide(p, best_row, is_best, emit_row, t_add, rel) -> np.ndarray:
    """Mask of emitted pairs under the rule, given per-row best p and emit flags."""
    return emit_row & (is_best | ((p >= t_add) & (p >= rel * best_row)))


def f05_counts(n_true: np.ndarray, n_pred: np.ndarray, tp: np.ndarray) -> np.ndarray:
    """Per-S1 F0.5 from counts; identical to metrics.f05_one."""
    f = np.zeros(len(n_true))
    f[(n_true == 0) & (n_pred == 0)] = 1.0
    ok = (n_true > 0) & (n_pred > 0) & (tp > 0)
    prec, rec = tp[ok] / n_pred[ok], tp[ok] / n_true[ok]
    f[ok] = 1.25 * prec * rec / (0.25 * prec + rec)
    return f


def grid_search(df: pd.DataFrame, eval_rows: np.ndarray, n_true: np.ndarray, grid: dict):
    """Evaluate every (t_emit, t_add, rel) on scored pairs of the S1 rows in eval_rows.

    `df` holds s1, p, label sorted by s1; `n_true` counts ALL true matches per eval S1
    (including matches blocking missed), so recall is never overestimated.
    """
    df = df[np.isin(df["s1"].to_numpy(), eval_rows)]  # only S1 records being evaluated
    s1, p, lab = df["s1"].to_numpy(), df["p"].to_numpy(), df["label"].to_numpy().astype(bool)
    starts, gid = groups(s1)
    best, is_best = best_of_group(p, starts, gid)
    best_row = best[gid]
    pos = np.searchsorted(eval_rows, s1[starts])  # eval position of each group
    rows = []
    for t_emit, t_add, rel in itertools.product(grid["t_emit"], grid["t_add"], grid["rel"]):
        mask = decide(p, best_row, is_best, (best >= t_emit)[gid], t_add, rel)
        n_pred = np.zeros(len(eval_rows))
        tp = np.zeros(len(eval_rows))
        n_pred[pos] = np.add.reduceat(mask.astype(np.int32), starts)
        tp[pos] = np.add.reduceat((mask & lab).astype(np.int32), starts)
        rows.append((round(t_emit, 2), round(t_add, 2), rel, f05_counts(n_true, n_pred, tp).mean()))
    table = pd.DataFrame(rows, columns=["t_emit", "t_add", "rel", "macro_f05"])
    return table.sort_values("macro_f05", ascending=False).reset_index(drop=True)


def assign_targets(s1: np.ndarray, t: np.ndarray, p: np.ndarray, margin: float = 0.0) -> np.ndarray:
    """Exclusivity (in the labels, every target belongs to at most one S1 record).

    Returns p with every pair zeroed except, for each target, its highest-probability S1;
    that one is zeroed too when it beats the runner-up S1 by less than `margin`.
    Needs the scores of all S1 candidates of each target (complete candidate graph).
    """
    order = np.lexsort((-p, t))
    ts, ps = t[order], p[order]
    first = np.r_[True, ts[1:] != ts[:-1]]
    second = np.zeros(len(ps), dtype=ps.dtype)
    has_next = np.r_[first[:-1] & ~first[1:], False]  # first rows whose target has 2+ S1
    second[np.flatnonzero(has_next)] = ps[np.flatnonzero(has_next) + 1]
    keep_sorted = first & ((ps - second) >= margin)
    keep = np.zeros(len(p), dtype=bool)
    keep[order] = keep_sorted
    return np.where(keep, p, 0.0).astype(p.dtype)


def expected_f_select(s1: np.ndarray, q: np.ndarray, miss: float = 0.0) -> np.ndarray:
    """Per-S1 set maximizing (approximate) expected F0.5, for rows sorted by s1.

    With calibrated q sorted descending inside a group, E[F] of predicting the top k is
    approximated by 1.25*S_k / (k + 0.25*(sum(q) + miss)), S_k the sum of the top-k q; the
    empty prediction scores P(no true match) = (1 - miss) * prod(1 - q). Choosing the best
    k is then a within-group argmax; the approximation implies "add the next candidate
    while its q exceeds 0.8 x the current expected F".
    """
    starts, gid = groups(s1)
    order = np.lexsort((-q, gid))
    qs = q[order]
    csum = np.cumsum(qs)
    base = np.r_[0.0, csum][starts][gid]            # cumsum before each group
    s_k = csum - base                                # sum of the top-k q (k = rank)
    k = np.arange(len(qs)) - starts[gid] + 1
    total = np.add.reduceat(qs, starts)[gid]
    f_k = 1.25 * s_k / (k + 0.25 * (total + miss))
    log_none = np.add.reduceat(np.log1p(-np.minimum(qs, 1 - 1e-9)), starts)
    f_none = (1 - miss) * np.exp(log_none)
    arg = np.lexsort((k, -f_k, gid))[starts]  # per group: highest expected F, smallest k on ties
    k_star = np.where(f_k[arg] > f_none, k[arg], 0)
    chosen_sorted = k <= k_star[gid]
    chosen = np.zeros(len(q), dtype=bool)
    chosen[order] = chosen_sorted
    return chosen


def apply_rule(df: pd.DataFrame, th: dict) -> np.ndarray:
    """Emitted-pair mask for scored pairs sorted by s1."""
    p = df["p"].to_numpy()
    starts, gid = groups(df["s1"].to_numpy())
    best, is_best = best_of_group(p, starts, gid)
    return decide(p, best[gid], is_best, (best >= th["t_emit"])[gid], th["t_add"], th["rel"])


def thresholds_path(cfg: dict):
    return lgbm_dir(cfg) / f"thresholds{cfg.get('stack', {}).get('rerank_tag', '')}.json"


def logit_adjust(q: np.ndarray, a: float, b: float) -> np.ndarray:
    """sigma(a * logit(q) + b): a temperature and shift on calibrated probabilities."""
    q = np.clip(q, 1e-6, 1 - 1e-6)
    return 1 / (1 + np.exp(-(a * np.log(q / (1 - q)) + b)))


def emit_mask(df: pd.DataFrame, params: dict) -> np.ndarray:
    """Emitted pairs (rows sorted by s1) under the configured selection method."""
    if params["method"] == "rule":
        return apply_rule(df, params)
    q = np.interp(df["p"].to_numpy(), params["iso_x"], params["iso_y"])
    q = logit_adjust(q, params["a"], params["b"])
    return expected_f_select(df["s1"].to_numpy(), q, params["miss"]) & (df["p"].to_numpy() > 0)


def score_mask(df, mask, eval_rows, n_true) -> float:
    """Macro F0.5 over eval_rows of the emitted pairs (vectorized f05_one)."""
    starts, _ = groups(df["s1"].to_numpy())
    pos = np.searchsorted(eval_rows, df["s1"].to_numpy()[starts])
    n_pred, tp = np.zeros(len(eval_rows)), np.zeros(len(eval_rows))
    n_pred[pos] = np.add.reduceat(mask.astype(np.int32), starts)
    tp[pos] = np.add.reduceat((mask & (df["label"].to_numpy() == 1)).astype(np.int32), starts)
    return float(f05_counts(n_true, n_pred, tp).mean())


def with_assignment(cfg: dict, frames: list[pd.DataFrame]) -> list[pd.DataFrame]:
    """Apply exclusivity across all frames (together they hold every S1 of the split)."""
    if not cfg["decide"].get("assignment"):
        return frames
    allp = pd.concat(frames, keys=range(len(frames)), names=["frame", "row"])
    p = assign_targets(allp["s1"].to_numpy(), allp["t"].to_numpy(), allp["p"].to_numpy(),
                       cfg["decide"].get("assignment_margin", 0.0))
    out, at = [], 0
    for f in frames:
        out.append(f.assign(p=p[at:at + len(f)]))
        at += len(f)
    return out


def run_tune(cfg: dict) -> dict:
    """Pick the decision parameters on validation; report macro F0.5, singleton acc, pair P/R.

    Optional steps (config `decide`): exclusivity across the complete train candidate graph,
    and expected-F0.5 selection on probabilities calibrated by isotonic regression fit on
    the training sample's out-of-fold predictions (never on validation).
    """
    s1 = load_normalized(cfg, "train", "s1", columns=["key"])
    tg = load_normalized(cfg, "train", "targets", columns=["key"])
    roles = s1_roles(cfg, s1["key"].to_numpy())
    eval_rows = np.flatnonzero(roles == 2)
    ts, tt = truth_pairs(cfg, s1["key"].to_numpy(), tg["key"].to_numpy())
    n_true = np.bincount(ts, minlength=len(s1))[eval_rows]
    pdir = preds_dir(cfg)
    stacked = bool(cfg.get("stack", {}).get("enabled"))
    if stacked and cfg["decide"].get("selection", "rule") != "rule":
        # stage 2 has its own target-competition features; expected-F would need a separate
        # calibration set
        raise SystemExit("stage 2 (stack.enabled) is supported with the rule decision only")
    tag = cfg.get("stack", {}).get("rerank_tag", "")
    if stacked and cfg["decide"].get("assignment"):
        # exclusivity on stage-2 p (in the labels, a target belongs to at most one S1). Needs the
        # stage-2 p of every train pair (stack.full_train_p), so each validation target's
        # competition is complete, as it is on test where every S1 is scored.
        full = pd.read_parquet(pdir / f"train_stack{tag}.parquet", columns=["s1", "t", "label", "p"])
        p_ex = assign_targets(full["s1"].to_numpy(), full["t"].to_numpy(), full["p"].to_numpy(),
                              cfg["decide"].get("assignment_margin", 0.0))
        log(f"assignment on stage-2 p: {int(((p_ex == 0) & (full['p'].to_numpy() > 0)).sum()):,} of "
            f"{len(full):,} train pairs zeroed (a target's non-best S1)")
        val = full.assign(p=p_ex)
        val = val[roles[val["s1"].to_numpy()] == 2]
        del full
    else:
        val = pd.read_parquet(pdir / (f"val_stack{tag}.parquet" if stacked else "val.parquet"),
                              columns=["s1", "t", "label", "p"])
    val = val.iloc[np.lexsort((val["t"].to_numpy(), val["s1"].to_numpy()))].reset_index(drop=True)
    others = []
    if not stacked:  # exclusivity / calibration need the other train pairs (stage-1 only)
        oof = pd.read_parquet(pdir / "train_oof.parquet", columns=["s1", "t", "label", "p"])
        others = [pd.read_parquet(pdir / "role0.parquet", columns=["s1", "t", "label", "p"])] \
            if (pdir / "role0.parquet").exists() else []
        if cfg["decide"].get("assignment") and not others:
            raise SystemExit("assignment needs the complete train graph (split.block_all_train_s1)")
        val, oof, *others = with_assignment(cfg, [val, oof] + others)
        oof = oof.sort_values(["s1", "t"], kind="stable")
        if not others:
            others = [oof]
    t0 = time.perf_counter()
    dcfg = cfg["decide"]
    if dcfg.get("selection", "rule") == "rule":
        table = grid_search(val, eval_rows, n_true, dcfg["grid"])
        best = table.iloc[0].to_dict()
        th = {"method": "rule", "t_emit": float(best["t_emit"]), "t_add": float(best["t_add"]),
              "rel": float(best["rel"])}
        top = table.head(5).to_dict(orient="records")
    else:
        from sklearn.isotonic import IsotonicRegression
        # calibrate on scores the model never trained on: role 0 when blocked (and always
        # with the single-fit model), else the 5-fold out-of-fold predictions
        pool = others[0]
        fit = pool.sample(n=min(len(pool), 5_000_000), random_state=cfg["seed"])
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(fit["p"], fit["label"])
        base = {"method": "expected_f", "iso_x": iso.X_thresholds_.tolist(), "iso_y": iso.y_thresholds_.tolist()}
        rows = []
        for miss, a, b in itertools.product(dcfg["ef_grid"]["miss"], dcfg["ef_grid"]["a"], dcfg["ef_grid"]["b"]):
            params = {**base, "miss": miss, "a": a, "b": b}
            rows.append((miss, a, b, score_mask(val, emit_mask(val, params), eval_rows, n_true)))
        table = pd.DataFrame(rows, columns=["miss", "a", "b", "macro_f05"]).sort_values("macro_f05", ascending=False)
        best = table.iloc[0].to_dict()
        th = {**base, "miss": float(best["miss"]), "a": float(best["a"]), "b": float(best["b"])}
        top = table.head(5).to_dict(orient="records")
    th["assignment"] = bool(dcfg.get("assignment"))
    th["assignment_margin"] = float(dcfg.get("assignment_margin", 0.0))
    th["stack"] = stacked  # these thresholds apply to stage-2 probabilities

    # Cross-check with the reference implementation of the metric.
    mask = emit_mask(val, th)
    in_eval = np.zeros(len(s1), bool)
    in_eval[eval_rows] = True
    truth = {int(r): set() for r in eval_rows}
    for a, b in zip(ts[in_eval[ts]], tt[in_eval[ts]]):
        truth[int(a)].add(int(b))
    pred = {}
    for a, b in zip(val["s1"].to_numpy()[mask], val["t"].to_numpy()[mask]):
        pred.setdefault(int(a), set()).add(int(b))
    score = macro_f05(truth, pred)
    assert abs(score - best["macro_f05"]) < 1e-9, (score, best["macro_f05"])
    prec, rec = pair_precision_recall(truth, pred)
    country = load_normalized(cfg, "train", "s1", columns=["country"])["country"].to_numpy(dtype=object)
    by_country = {}
    for c in pd.unique(country[eval_rows]):
        keys = [int(r) for r in eval_rows[country[eval_rows] == c]]
        by_country[c] = round(macro_f05({k: truth[k] for k in keys}, pred), 5)
    report = {"thresholds": th, "val_macro_f05": round(score, 5), "val_by_country": by_country,
              "singleton_accuracy": round(singleton_accuracy(truth, pred), 5),
              "pair_precision": round(prec, 5), "pair_recall": round(rec, 5),
              "val_s1": len(eval_rows), "grid_size": len(table),
              "grid_seconds": round(time.perf_counter() - t0, 1), "top5": top}
    if stacked and cfg["stack"].get("rerank"):  # where the validation errors are (next-step diagnostics)
        from .stack import rerank_blocks
        rr_tag = rerank_blocks(cfg)[0][1]  # the first reranker's scores ("_rawboth" has no file of its own)
        rr = pd.read_parquet(cfg["paths"]["artifacts_dir"] / "rerank" / f"scores_train{rr_tag}.parquet",
                             columns=["s1", "t"])
        n_t = int(max(rr["t"].max(), val["t"].max())) + 1
        has = np.isin(val["s1"].to_numpy().astype(np.int64) * n_t + val["t"].to_numpy(),
                      rr["s1"].to_numpy().astype(np.int64) * n_t + rr["t"].to_numpy())
        lab = val["label"].to_numpy() == 1
        report["errors"] = {"true_pairs_in_candidates": int(lab.sum()), "true_pairs_not_reranked": int((lab & ~has).sum()),
                            "missed_true_pairs": int((lab & ~mask).sum()),
                            "missed_true_pairs_not_reranked": int((lab & ~mask & ~has).sum()),
                            "false_pairs_emitted": int((~lab & mask).sum()),
                            "false_pairs_emitted_not_reranked": int((~lab & mask & ~has).sum())}
        log(f"validation errors: {report['errors']}")
    with open(thresholds_path(cfg), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    shown = {k: v for k, v in th.items() if k not in ("iso_x", "iso_y")}
    log(f"tuned decision {shown}: val macro F0.5 {score:.5f} {by_country}, singleton acc "
        f"{report['singleton_accuracy']:.4f}, pair P {prec:.4f} R {rec:.4f}")
    return report


def run_predict(cfg: dict) -> None:
    """Score test candidates, apply the tuned rule, write both outputs and validate them."""
    from .train import score_test

    t_start = time.perf_counter()
    with open(thresholds_path(cfg), encoding="utf-8") as f:
        th = json.load(f)["thresholds"]
    th.setdefault("method", "rule")
    if bool(th.get("stack")) != bool(cfg.get("stack", {}).get("enabled")):
        raise SystemExit("thresholds.json and stack.enabled disagree; re-run the train (or stack) stage")
    scored = score_test(cfg)  # stage-1 p (cached in preds/test_stage1.parquet), sorted by (s1, t)
    tag = cfg.get("stack", {}).get("rerank_tag", "")
    if th.get("stack"):  # stage 2 re-scores every test pair from the stage-1 candidate graph
        import lightgbm as lgb
        from . import stack
        from .utils import n_threads
        scored = stack.attach_rerank(cfg, scored, "test")  # reranker probabilities (stack.rerank)
        scored["p1"] = scored["p"]
        scored["p"] = stack.rescore(lgb.Booster(model_file=str(stack.model_path(cfg))), scored,
                                    stack.target_data(cfg, "test"), n_threads(cfg))
    keep_cols = [c for c in ("s1", "t", "p", "p1", "rr_p") if c in scored]
    scored[keep_cols].to_parquet(preds_dir(cfg) / f"test{tag}.parquet", index=False)
    if th.get("assignment"):  # every test S1 was blocked, so the target competition is complete
        scored["p"] = assign_targets(scored["s1"].to_numpy(), scored["t"].to_numpy(),
                                     scored["p"].to_numpy(), th.get("assignment_margin", 0.0))
    mask = emit_mask(scored, th)
    log(f"test: {len(scored):,} scored pairs, {int(mask.sum()):,} emitted")
    test_c = load_normalized(cfg, "test", "s1", columns=["country"])["country"].to_numpy(dtype=object)
    report = {"test": emission_profile(scored, mask, test_c)}
    val_path = preds_dir(cfg) / (f"val_stack{tag}.parquet" if th.get("stack") else "val.parquet")
    if val_path.exists():  # the same profile on validation, for comparison with unseen countries
        val = pd.read_parquet(val_path, columns=["s1", "t", "p"])
        val = val.iloc[np.lexsort((val["t"].to_numpy(), val["s1"].to_numpy()))].reset_index(drop=True)
        train_c = load_normalized(cfg, "train", "s1", columns=["country"])["country"].to_numpy(dtype=object)
        report["validation"] = emission_profile(val, emit_mask(val, th), train_c)
    (preds_dir(cfg) / f"emission_profile{tag}.json").write_text(json.dumps(report, indent=1))
    log("emission profile by country (empty share, matches per matched S1, p of emitted): "
        + json.dumps(report))

    s1_ids = load_source(cfg, "test", 1, columns=["entity_id"])["entity_id"].to_numpy(dtype=object)
    t_ids = np.concatenate([load_source(cfg, "test", n, columns=["entity_id"])["entity_id"]
                            .to_numpy(dtype=object) for n in (2, 3)])
    s, t = scored["s1"].to_numpy(), scored["t"].to_numpy()
    bounds = np.searchsorted(s, np.arange(len(s1_ids) + 1))  # rows of S1 i: bounds[i]:bounds[i+1]
    cand_lists = [t_ids[t[bounds[i]:bounds[i + 1]]].tolist() for i in range(len(s1_ids))]
    sm, tm = s[mask], t[mask]
    mb = np.searchsorted(sm, np.arange(len(s1_ids) + 1))
    match_lists = [t_ids[tm[mb[i]:mb[i + 1]]].tolist() for i in range(len(s1_ids))]

    # Invariants: one row per test S1, no duplicate IDs, matches are a subset of candidates.
    assert len(set(s1_ids)) == len(s1_ids)
    assert all(len(c) == len(set(c)) for c in cand_lists)
    assert all(set(m) <= set(c) for m, c in zip(match_lists, cand_lists) if m)
    out = cfg["paths"]["output_dir"]
    write_id_lists(out / "matching_results.tsv", ["source1_entity_id", "matched_entity_ids"],
                   s1_ids.tolist(), match_lists)
    write_id_lists(out / "candidate_pairs.tsv", ["source1_entity_id", "candidate_entity_ids"],
                   s1_ids.tolist(), cand_lists)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    archive = cfg["paths"]["artifacts_dir"] / "submissions" / stamp
    archive.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out / "matching_results.tsv", archive / "matching_results.tsv")
    shutil.copy2(out / "candidate_pairs.tsv", archive / "candidate_pairs.tsv")  # zip needs both of one run
    with open(archive / "thresholds.json", "w", encoding="utf-8") as f:
        json.dump(th, f)
    n_matched = sum(bool(m) for m in match_lists)
    log(f"wrote outputs ({n_matched:,} of {len(s1_ids):,} S1 with matches, "
        f"{int(mask.sum()):,} matched IDs); archived to {archive}")

    # The official validator keeps every candidate ID in Python sets (~15 GB for ~140M
    # candidate IDs), so it checks the scored file with --check-ids, and the candidate file
    # gets the same rules from a streaming check.
    from .utils import REPO_ROOT
    validator = REPO_ROOT.parents[1] / "utils" / "validate_submission.py"  # the challenge's validator
    result = subprocess.run([sys.executable, str(validator), "--matching", str(out / "matching_results.tsv"),
                             "--candidate", str(out / "no_candidate_file_given.tsv"),
                             "--test-dir", str(cfg["paths"]["data_dir"] / "test"), "--check-ids"],
                            capture_output=True, text=True, encoding="utf-8")
    print(result.stdout[-3000:], result.stderr[-2000:])
    problems = check_candidate_file(out / "candidate_pairs.tsv", out / "matching_results.tsv",
                                    set(s1_ids.tolist()), set(t_ids.tolist()))
    print("candidate_pairs.tsv streaming check:", "PASS" if not problems else problems)
    log(f"predict done in {time.perf_counter() - t_start:.0f}s; validator exit code {result.returncode}")


def emission_profile(df: pd.DataFrame, mask: np.ndarray, country: np.ndarray) -> dict:
    """Per country of the S1 rows in df (sorted by s1): S1 count, share with no emitted pair,
    emitted pairs per matched S1, mean best p and mean p of emitted pairs."""
    s1, p = df["s1"].to_numpy(), df["p"].to_numpy()
    starts, _ = groups(s1)
    n_emit = np.add.reduceat(mask.astype(np.int32), starts)
    best = np.maximum.reduceat(p, starts)
    g_c = country[s1[starts]]
    emitted_c = country[s1[mask]]
    out = {}
    for c in pd.unique(g_c):
        m = g_c == c
        matched = n_emit[m] > 0
        out[str(c)] = {"s1": int(m.sum()), "empty_share": round(float(1 - matched.mean()), 4),
                       "per_matched_s1": round(float(n_emit[m][matched].mean()), 3) if matched.any() else 0.0,
                       "mean_best_p": round(float(best[m].mean()), 4),
                       "mean_emitted_p": round(float(p[mask][emitted_c == c].mean()), 4) if matched.any() else 0.0}
    return out


def check_candidate_file(cand_path, match_path, s1_ids: set, target_ids: set) -> list[str]:
    """Streaming version of the validator's rules for candidate_pairs.tsv.

    Checks header, one row per test S1 (no duplicates, none missing), S2/S3-only IDs that
    exist in the test files, no duplicates within a list, and matches subset of candidates.
    """
    matches = {}
    with open(match_path, encoding="utf-8") as f:
        next(f)
        for line in f:
            s1, _, rest = line.rstrip("\n").partition("\t")
            matches[s1] = set(rest.split(",")) if rest else set()
    problems, seen = [], set()
    with open(cand_path, encoding="utf-8") as f:
        if f.readline().rstrip("\n").split("\t") != ["source1_entity_id", "candidate_entity_ids"]:
            problems.append("bad header")
        for n, line in enumerate(f, start=2):
            s1, tab, rest = line.rstrip("\n").partition("\t")
            if not tab:
                problems.append(f"line {n}: no tab")
                continue
            if s1 in seen:
                problems.append(f"duplicate row {s1}")
            seen.add(s1)
            ids = rest.split(",") if rest else []
            id_set = set(ids)
            if len(ids) != len(id_set):
                problems.append(f"repeated ID in list of {s1}")
            if any(not i.startswith(("S2-", "S3-")) or i not in target_ids for i in ids):
                problems.append(f"bad or unknown ID in list of {s1}")
            if not matches.get(s1, set()) <= id_set:
                problems.append(f"matches of {s1} not a subset of its candidates")
            if len(problems) > 20:
                break
    if seen != s1_ids:
        problems.append(f"{len(s1_ids - seen)} test S1 missing, {len(seen - s1_ids)} unknown S1 rows")
    return problems
