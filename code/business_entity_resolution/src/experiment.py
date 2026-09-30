"""Quick-protocol experiments for stage 1 (one change at a time).

Protocol, cheaper than the full 5-fold train stage but identical for every experiment:
* fit: one GroupKFold split of the training sample (80% of its S1 records), early
  stopping on the other 20%;
* validation: score every validation pair, grid-search the decision rule on validation
  macro F0.5, report singleton accuracy and pair precision/recall;
* LOCO (France proxy): train on one country's training rows, tune the rule on that
  country's validation rows, score the other country's validation rows.

Usage:  python -m src.experiment --tag lr005 --set lgbm.params.learning_rate=0.05 [--drop a,b] [--loco]
Results are appended to artifacts/experiments/results.jsonl.

Fits and their scores are cached under artifacts/experiments/cache/<key>/, the key hashing
the feature list, the LightGBM settings and the feature files, so experiments that differ
only in the decision step (assignment, expected-F) reuse one fit.
"""

import argparse
import hashlib
import json
import os
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

from .decide import apply_rule, f05_counts, groups, grid_search  # noqa: E402
from .features import FEATURES_VERSION  # noqa: E402
from .io import s1_roles, truth_pairs  # noqa: E402
from .normalize.normalize import load_normalized  # noqa: E402
from .train import extra_columns, feature_parts, lgb_params  # noqa: E402
from .utils import load_config, log, n_threads  # noqa: E402


def set_value(cfg: dict, dotted: str, value: str) -> None:
    """Apply a --set override like lgbm.params.learning_rate=0.05 (value parsed as JSON)."""
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    try:
        node[keys[-1]] = json.loads(value)
    except json.JSONDecodeError:
        node[keys[-1]] = value


def eval_parts(cfg) -> list:
    """Feature parts restricted to the training sample and validation S1 (roles 1 and 2).

    With every train S1 blocked, role-0 rows are ~75% of the train features, and the quick
    protocol's fits and validation scoring need only roles 1 and 2, so they read these
    smaller copies (rebuilt whenever the feature parts change).
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    parts = feature_parts(cfg, "train")
    from .features import features_dir
    d = cfg["paths"]["artifacts_dir"] / "experiments" / "subset" / features_dir(cfg, "train").name
    stamp = json.dumps([(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in parts])
    if (d / "stamp.json").exists() and (d / "stamp.json").read_text() == stamp:
        return sorted(d.glob("part-*.parquet"))
    d.mkdir(parents=True, exist_ok=True)
    (d / "stamp.json").unlink(missing_ok=True)
    for old in d.glob("part-*.parquet"):
        old.unlink()
    t0 = time.perf_counter()
    for p in parts:
        table = pq.read_table(p)
        table = table.filter(pc.is_in(table["role"], value_set=pa.array([1, 2], table["role"].type)))
        pq.write_table(table, d / p.name)
        del table
    (d / "stamp.json").write_text(stamp)
    log(f"roles 1-2 feature subset written in {time.perf_counter() - t0:.0f}s")
    return sorted(d.glob("part-*.parquet"))


def load_rows(cfg, names, role, s1_country=None, country=None):
    """Feature matrix, labels and S1 ids of one role (optionally one country), preallocated."""
    parts = eval_parts(cfg) if role in (1, 2) else feature_parts(cfg, "train")
    masks = []
    for p in parts:
        meta = pd.read_parquet(p, columns=["s1", "role"])
        m = meta["role"].to_numpy() == role
        if country is not None:
            m &= s1_country[meta["s1"].to_numpy()] == country
        masks.append(m)
    X = np.empty((sum(int(m.sum()) for m in masks), len(names)), dtype=np.float32)
    y, s, at = [], [], 0
    for p, m in zip(parts, masks):
        df = pd.read_parquet(p, columns=names + ["s1", "label"])[m]
        X[at:at + len(df)] = df[names].to_numpy(np.float32)
        at += len(df)
        y.append(df["label"].to_numpy())
        s.append(df["s1"].to_numpy())
    return X, np.concatenate(y), np.concatenate(s)


def fit(cfg, names, s1_country=None, country=None) -> lgb.Booster:
    """One GroupKFold split of the training sample (optionally one country): train on 4/5
    of the S1 groups, early-stop on the rest.

    The data are binned once and both parts are Dataset subsets (no copies of X); the float
    matrix is loaded here so that it is freed once binned.
    """
    X, y, s = load_rows(cfg, names, 1, s1_country, country)
    tr, es = next(GroupKFold(n_splits=5).split(np.zeros((len(y), 1)), y, s))
    full = lgb.Dataset(X, y, feature_name=names, free_raw_data=True).construct()
    del X, s
    lcfg = cfg["lgbm"]
    return lgb.train(lgb_params(cfg, names), full.subset(np.sort(tr)), num_boost_round=lcfg["max_rounds"],
                     valid_sets=[full.subset(np.sort(es))], valid_names=["es"],
                     callbacks=[lgb.early_stopping(lcfg["early_stopping"], first_metric_only=True,
                                                   verbose=False),
                                lgb.log_evaluation(lcfg["log_every"])])


def score_val(cfg, booster, names, roles=(2,)) -> pd.DataFrame:
    """Probabilities (s1, t, label, role, p, and the stage-2 pair features) for the pairs of
    the given S1 roles, sorted by (s1, t). Compact dtypes (int32/int8/float32)."""
    import pyarrow.parquet as pq

    parts = eval_parts(cfg) if set(roles) <= {1, 2} else feature_parts(cfg, "train")
    extras = extra_columns(cfg, pq.read_schema(parts[0]).names)
    cols = {k: [] for k in ["s1", "t", "label", "role", "p"] + extras}
    for path in parts:
        df = pd.read_parquet(path, columns=list(dict.fromkeys(names + extras + ["s1", "t", "label", "role"])))
        df = df[df["role"].isin(roles)]
        cols["s1"].append(df["s1"].to_numpy(np.int32))
        cols["t"].append(df["t"].to_numpy(np.int32))
        cols["label"].append(df["label"].to_numpy(np.int8))
        cols["role"].append(df["role"].to_numpy(np.int8))
        cols["p"].append(booster.predict(df[names].to_numpy(np.float32), num_iteration=booster.best_iteration,
                                         num_threads=n_threads(cfg)).astype(np.float32))
        for c in extras:
            cols[c].append(df[c].to_numpy(np.float32))
        del df
    arr = {}
    for k in list(cols):
        arr[k] = np.concatenate(cols.pop(k))  # frees each column's part list as it goes
    order = np.lexsort((arr["t"], arr["s1"]))
    return pd.DataFrame({k: arr.pop(k)[order] for k in list(arr)})


def model_key(cfg, names) -> str:
    """Identity of a quick-protocol fit: feature list and data, LightGBM settings."""
    parts = feature_parts(cfg, "train")
    params = {k: v for k, v in lgb_params(cfg, names).items() if k != "num_threads"}
    ident = {"names": names, "params": params, "rounds": cfg["lgbm"]["max_rounds"],
             "early_stopping": cfg["lgbm"]["early_stopping"], "features_version": FEATURES_VERSION,
             "parts": [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in parts]}
    return hashlib.sha1(json.dumps(ident, sort_keys=True, default=str).encode()).hexdigest()[:16]


def cached_fit(cfg, names, s1_country=None, country=None):
    """fit() on the training sample (optionally one country), cached with its best iteration.

    The saved model keeps only the best iteration's trees, and a loaded booster predicts
    with all of them, so cached and fresh fits score identically.
    """
    d = cfg["paths"]["artifacts_dir"] / "experiments" / "cache" / (
        model_key(cfg, names) + ("" if country is None else f"-{country}"))
    if (d / "fit.json").exists():
        info = json.loads((d / "fit.json").read_text())
        log(f"cached fit {d.name} (best iteration {info['best_iteration']})")
        return lgb.Booster(model_file=str(d / "booster.txt")), info["best_iteration"], d
    booster = fit(cfg, names, s1_country, country)
    d.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(d / "booster.partial"), num_iteration=booster.best_iteration)
    (d / "booster.partial").replace(d / "booster.txt")
    (d / "fit.json").write_text(json.dumps({"best_iteration": booster.best_iteration, "names": names}))
    return booster, booster.best_iteration, d


def cached_scores(cfg, booster, names, d, roles=(2,)) -> pd.DataFrame:
    """score_val() cached next to its fit."""
    tag = hashlib.sha1(json.dumps(extra_columns(cfg, names + ["emb_cos"])).encode()).hexdigest()[:6]
    path = d / f"scores_role{''.join(map(str, roles))}_{tag}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    df = score_val(cfg, booster, names, roles)
    df.to_parquet(path.with_name(path.name + ".partial"), index=False)
    path.with_name(path.name + ".partial").replace(path)
    return df


def decision_eval(cfg, scored: pd.DataFrame, val_rows, n_true_all) -> tuple[dict, dict]:
    """Tune the configured decision (assignment, rule or expected-F) on validation.

    `scored` holds every scored train pair; with assignment on it must include the other
    S1 roles, so each target's competitors are present. Role-1 pairs are in-sample for the
    quick model (their scores are optimistic); the full train stage uses out-of-fold scores.
    """
    from .decide import assign_targets, emit_mask, score_mask
    import itertools
    from sklearn.isotonic import IsotonicRegression

    d = cfg["decide"]
    p, role, label = scored["p"].to_numpy(), scored["role"].to_numpy(), scored["label"].to_numpy()
    if d.get("assignment"):
        p = assign_targets(scored["s1"].to_numpy(), scored["t"].to_numpy(), p, d.get("assignment_margin", 0.0))
    is_val = role == 2
    val = pd.DataFrame({"s1": scored["s1"].to_numpy()[is_val], "t": scored["t"].to_numpy()[is_val],
                        "label": label[is_val], "p": p[is_val]})  # still sorted by s1
    n_true = n_true_all[val_rows]
    if d.get("selection", "rule") == "rule":
        table = grid_search(val, val_rows, n_true, d["grid"])
        params = {"method": "rule", **{k: float(table.iloc[0][k]) for k in ("t_emit", "t_add", "rel")}}
    else:
        # calibrate on out-of-sample scores: role 0 (never trained on) when blocked, else role 1
        pool = np.flatnonzero(role == 0) if (role == 0).any() else np.flatnonzero(role == 1)
        rng = np.random.default_rng(cfg["seed"])
        pick = np.sort(rng.choice(pool, size=min(len(pool), 5_000_000), replace=False))
        del pool
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p[pick], label[pick])
        base = {"method": "expected_f", "iso_x": iso.X_thresholds_, "iso_y": iso.y_thresholds_}
        best = max(((score_mask(val, emit_mask(val, {**base, "miss": m, "a": a, "b": b}), val_rows, n_true), m, a, b)
                    for m, a, b in itertools.product(d["ef_grid"]["miss"], d["ef_grid"]["a"], d["ef_grid"]["b"])),
                   key=lambda x: x[0])
        params = {**base, "miss": best[1], "a": best[2], "b": best[3]}
    mask = emit_mask(val, params)
    shown = {k: v for k, v in params.items() if k not in ("iso_x", "iso_y")}
    return {"macro_f05": round(score_mask(val, mask, val_rows, n_true), 5)}, shown


def stack_eval(cfg, booster, cache, names, roles, val_rows, n_true_all, country, src=None) -> dict:
    """Stage 2 (src/stack.py) on top of a quick stage-1 fit.

    Stage 1 scores every train pair (the target competition needs all S1 roles); stage 2
    trains on a seeded role-0 S1 sample (of country `src` for LOCO) and the rule is tuned
    on validation (src's validation for LOCO). Returns validation metrics per country with
    those thresholds.
    """
    from . import stack

    scored = stack.attach_rerank(cfg, cached_scores(cfg, booster, names, cache, roles=(0, 1, 2, 3)), "train")
    s1_all = scored["s1"].to_numpy()
    comp = stack.competition(scored)
    tgt = stack.target_data(cfg, "train")
    tr = stack.train_rows(cfg, scored, roles, country, src)
    b2 = stack.cached_fit(cfg, cache if src is None else None,  # the main stage 2 is reused by the pipeline
                          lambda: stack.features_for(scored, tr, comp, tgt, stack.feature_list(cfg)),
                          lambda: scored["label"].to_numpy()[tr], lambda: s1_all[tr])
    vr = np.flatnonzero(roles[s1_all] == 2)
    p2 = stack.predict_rows(b2, scored, vr, comp, tgt)
    val2 = pd.DataFrame({"s1": s1_all[vr], "t": scored["t"].to_numpy()[vr],
                         "label": scored["label"].to_numpy()[vr], "p": p2})
    val1 = val2.assign(p=scored["p"].to_numpy()[vr])  # stage 1 under the same protocol
    del scored, comp
    tune_rows = val_rows if src is None else val_rows[country[val_rows] == src]
    out = {"best_iteration": b2.current_iteration()}  # the saved/returned model holds the best iteration
    for name, frame in (("stage2", val2), ("stage1", val1)):
        t = grid_search(frame, tune_rows, n_true_all[tune_rows], cfg["decide"]["grid"]).iloc[0]
        th = {k: float(t[k]) for k in ("t_emit", "t_add", "rel")}
        res = {"thresholds": th, **evaluate(frame, val_rows, n_true_all[val_rows], th),
               "by_country": {c: evaluate(frame, val_rows[country[val_rows] == c],
                                          n_true_all[val_rows[country[val_rows] == c]], th)["macro_f05"]
                              for c in pd.unique(country[val_rows])}}
        if name == "stage2":
            out.update(res)
        else:
            out["stage1_same_protocol"] = res
    return out


def evaluate(df, eval_rows, n_true, th) -> dict:
    """Macro F0.5, singleton accuracy and pair P/R of a fixed rule on the eval S1 rows."""
    df = df[np.isin(df["s1"].to_numpy(), eval_rows)]
    mask = apply_rule(df, th)
    starts, _ = groups(df["s1"].to_numpy())
    pos = np.searchsorted(eval_rows, df["s1"].to_numpy()[starts])
    n_pred, tp = np.zeros(len(eval_rows)), np.zeros(len(eval_rows))
    n_pred[pos] = np.add.reduceat(mask.astype(np.int32), starts)
    tp[pos] = np.add.reduceat((mask & (df["label"].to_numpy() == 1)).astype(np.int32), starts)
    f = f05_counts(n_true, n_pred, tp)
    single = n_true == 0
    return {"macro_f05": round(float(f.mean()), 5),
            "singleton_acc": round(float((n_pred[single] == 0).mean()), 4),
            "pair_precision": round(float(tp.sum() / max(n_pred.sum(), 1)), 4),
            "pair_recall": round(float(tp.sum() / max(n_true.sum(), 1)), 4)}


def main(argv=None) -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--set", action="append", default=[], help="dotted.config.key=json_value")
    ap.add_argument("--drop", default="", help="comma-separated features to leave out")
    ap.add_argument("--loco", action="store_true", help="also run both leave-one-country-out directions")
    ap.add_argument("--stack", action="store_true", help="also evaluate stage 2 (src/stack.py) on this fit")
    args = ap.parse_args(argv)

    cfg = load_config()
    for item in args.set:
        key, _, value = item.partition("=")
        set_value(cfg, key, value)
    t0 = time.perf_counter()
    s1 = load_normalized(cfg, "train", "s1", columns=["key", "country"])
    tg = load_normalized(cfg, "train", "targets", columns=["key"])
    roles = s1_roles(cfg, s1["key"].to_numpy())
    ts, _ = truth_pairs(cfg, s1["key"].to_numpy(), tg["key"].to_numpy())
    n_true_all = np.bincount(ts, minlength=len(s1))
    country = s1["country"].to_numpy(dtype=object)
    from fnmatch import fnmatch
    import pyarrow.parquet as pq
    columns = pq.read_schema(feature_parts(cfg, "train")[0]).names
    patterns = list(filter(None, args.drop.split(","))) + list(cfg["lgbm"].get("drop_features", []))
    features = [c for c in columns if c not in {"s1", "t", "label", "role"}]
    drop = {c for c in features if any(fnmatch(c, p) for p in patterns)}  # names or wildcards
    names = [c for c in features if c not in drop]
    val_rows = np.flatnonzero(roles == 2)
    grid = cfg["decide"]["grid"]

    booster, best_it, cache = cached_fit(cfg, names)
    val = cached_scores(cfg, booster, names, cache)
    table = grid_search(val, val_rows, n_true_all[val_rows], grid)
    th = {k: float(table.iloc[0][k]) for k in ("t_emit", "t_add", "rel")}
    result = {"tag": args.tag, "set": args.set, "drop": sorted(drop), "n_features": len(names),
              "best_iteration": best_it, "thresholds": th,
              **evaluate(val, val_rows, n_true_all[val_rows], th),
              "by_country": {c: evaluate(val, val_rows[country[val_rows] == c],
                                         n_true_all[val_rows[country[val_rows] == c]], th)["macro_f05"]
                             for c in pd.unique(country[val_rows])}}
    log(f"{args.tag}: val macro F0.5 {result['macro_f05']} {result['by_country']} "
        f"(best iteration {best_it})")
    d = cfg["decide"]
    if d.get("assignment") or d.get("selection", "rule") != "rule":
        scored = cached_scores(cfg, booster, names, cache, roles=(0, 1, 2, 3))
        metrics, params = decision_eval(cfg, scored, val_rows, n_true_all)
        result["decision"] = {**metrics, "params": params}
        log(f"{args.tag}: with decision {params}: val macro F0.5 {metrics['macro_f05']}")
        del scored
    if args.stack:
        result["stack"] = stack_eval(cfg, booster, cache, names, roles, val_rows, n_true_all, country)
        log(f"{args.tag}: stage 2: val macro F0.5 {result['stack']['macro_f05']} {result['stack']['by_country']}")

    if args.loco:
        result["loco"] = {}
        if args.stack:
            result["stack_loco"] = {}
        for src in pd.unique(country[val_rows]):
            b, _, c_cache = cached_fit(cfg, names, country, src)
            v = cached_scores(cfg, b, names, c_cache)
            src_rows = val_rows[country[val_rows] == src]
            t = grid_search(v, src_rows, n_true_all[src_rows], grid).iloc[0]
            th_src = {k: float(t[k]) for k in ("t_emit", "t_add", "rel")}
            for dst in pd.unique(country[val_rows]):
                if dst != src:
                    dst_rows = val_rows[country[val_rows] == dst]
                    result["loco"][f"{src}->{dst}"] = evaluate(v, dst_rows, n_true_all[dst_rows],
                                                               th_src)["macro_f05"]
            del v
            if args.stack:
                # Approximate LOCO for stage 2 only: the main (both-country) stage-1 scores,
                # stage 2 trained on src's role-0 S1 and the rule tuned on src's validation.
                # (A full LOCO would re-score all 125M train pairs per direction, ~20 min each.)
                s2 = stack_eval(cfg, booster, cache, names, roles, val_rows, n_true_all, country, src)
                for dst, f in s2["by_country"].items():
                    if dst != src:
                        result["stack_loco"][f"{src}->{dst}"] = {
                            "stage2": f, "stage1": s2["stage1_same_protocol"]["by_country"][dst]}
        log(f"{args.tag}: LOCO {result['loco']}" + (f"; stage 2 LOCO {result['stack_loco']}" if args.stack else ""))
    result["minutes"] = round((time.perf_counter() - t0) / 60, 1)
    out = cfg["paths"]["artifacts_dir"] / "experiments"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
