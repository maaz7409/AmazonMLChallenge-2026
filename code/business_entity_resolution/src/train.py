"""Model 1: LightGBM pair scorer.

Trained from scratch on the candidate pairs of the training-sample S1 records (role 1; all
negatives are hard negatives from blocking). `lgbm.folds: 1` (the final setting, run_single):
one fit on 4/5 of the sample's S1 groups, early-stopped on the other fifth, which scores every
pair. `lgbm.folds: k > 1`: GroupKFold(k) by S1 gives out-of-fold probabilities for every
training pair, and validation and test pairs get the average of the k fold models.

Memory: features are copied once into a preallocated float32 matrix, LightGBM bins them
once, and folds are Dataset.subset views, so no per-fold copies of the data are made.
"""

import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .features import features_dir
from .utils import log, n_threads

NON_FEATURES = {"s1", "t", "label", "role"}


def lgbm_dir(cfg: dict):
    return cfg["paths"]["models_dir"] / "lgbm"


def preds_dir(cfg: dict):
    return cfg["paths"]["artifacts_dir"] / "preds"


def feature_parts(cfg: dict, split: str) -> list:
    return sorted(features_dir(cfg, split).glob("part-*.parquet"))


def feature_names(cfg: dict, split: str = "train") -> list[str]:
    """Model inputs: every feature column except identifiers, labels and dropped ones
    (lgbm.drop_features accepts exact names or wildcards such as "*_pc_*")."""
    from fnmatch import fnmatch
    columns = pd.read_parquet(feature_parts(cfg, split)[0]).columns
    patterns = cfg["lgbm"].get("drop_features", [])
    return [c for c in columns if c not in NON_FEATURES and not any(fnmatch(c, p) for p in patterns)]


def load_matrix(cfg: dict, split: str, role: int | None, names: list[str]):
    """(X float32, label or None, s1, t) for all pairs of a split, optionally one S1 role."""
    parts = feature_parts(cfg, split)
    masks = [None if role is None else pd.read_parquet(p, columns=["role"])["role"].to_numpy() == role
             for p in parts]
    total = sum(len(pd.read_parquet(p, columns=["s1"])) if m is None else int(m.sum())
                for p, m in zip(parts, masks))
    X = np.empty((total, len(names)), dtype=np.float32)
    y, s1, t = [], [], []
    at = 0
    for p, m in zip(parts, masks):
        df = pd.read_parquet(p, columns=names + ["s1", "t"] + (["label"] if split == "train" else []))
        if m is not None:
            df = df[m]
        X[at:at + len(df)] = df[names].to_numpy(np.float32)
        at += len(df)
        s1.append(df["s1"].to_numpy())
        t.append(df["t"].to_numpy())
        if "label" in df:
            y.append(df["label"].to_numpy())
    return X, (np.concatenate(y) if y else None), np.concatenate(s1), np.concatenate(t)


def monotone_constraints(names: list[str]) -> list[int]:
    """Domain-knowledge direction of each feature's effect on the match probability.

    +1: similarities, shared-token evidence, agreement states, blocker scores, margins.
    -1: ranks and gaps behind the best candidate, rarest unmatched token, token-count gap.
     0: counts, frequencies, flags and anything else (left free).
    """
    def sign(n: str) -> int:
        if n.endswith(("_rank", "_grank", "_gap", "_token_diff")) or "_max_unmatched_idf" in n:
            return -1
        if n.endswith(("_margin", "_score", "_ratio", "_tset", "_tsort", "_partial", "_jw", "_lev",
                       "_idf_jacc", "_idf_cover_s1", "_idf_cover_t", "_max_shared_idf",
                       "_n_shared_rare", "_n_shared", "_eq")) or n in ("combo", "acronym_match"):
            return 1
        return 0
    return [sign(n) for n in names]


def lgb_params(cfg: dict, names: list[str] | None = None) -> dict:
    """LightGBM parameters: config params plus fixed objective/seed settings.

    With lgbm.monotone set, domain-knowledge monotone constraints are added (they need the
    feature names); intended to make the model transfer better across countries.
    """
    p = dict(cfg["lgbm"]["params"])
    p.update(objective="binary", metric=["binary_logloss", "auc"], seed=cfg["seed"],
             deterministic=True, force_row_wise=True, verbose=-1, num_threads=n_threads(cfg))
    if cfg["lgbm"].get("monotone") and names is not None:
        p.update(monotone_constraints=monotone_constraints(names), monotone_constraints_method="advanced")
    return p


def fit_folds(cfg: dict, X: np.ndarray, y: np.ndarray, groups: np.ndarray, names: list[str],
              n_rounds: int | None = None) -> tuple[list[lgb.Booster], np.ndarray]:
    """GroupKFold by S1: one early-stopped booster per fold plus out-of-fold probabilities."""
    lcfg = cfg["lgbm"]
    full = lgb.Dataset(X, y, feature_name=names, free_raw_data=False).construct()
    oof = np.zeros(len(y), dtype=np.float32)
    boosters = []
    splitter = GroupKFold(n_splits=lcfg["folds"])
    for fold, (tr, va) in enumerate(splitter.split(np.zeros((len(y), 1)), y, groups)):
        t0 = time.perf_counter()
        booster = lgb.train(lgb_params(cfg, names), full.subset(np.sort(tr)),
                            num_boost_round=n_rounds or lcfg["max_rounds"],
                            valid_sets=[full.subset(np.sort(va))], valid_names=["fold_valid"],
                            callbacks=[lgb.early_stopping(lcfg["early_stopping"], first_metric_only=True,
                                                          verbose=False),  # binary_logloss
                                       lgb.log_evaluation(lcfg["log_every"])])
        oof[va] = booster.predict(X[va], num_iteration=booster.best_iteration)
        boosters.append(booster)
        log(f"fold {fold}: best iteration {booster.best_iteration}, "
            f"auc {booster.best_score['fold_valid']['auc']:.5f}, {time.perf_counter() - t0:.0f}s")
    return boosters, oof


def predict(boosters: list[lgb.Booster], X: np.ndarray, threads: int = 0) -> np.ndarray:
    """Average probability of the fold models (explicit thread count: see utils.n_threads)."""
    kw = {"num_threads": threads} if threads else {}
    return np.mean([b.predict(X, num_iteration=b.best_iteration, **kw) for b in boosters], axis=0
                   ).astype(np.float32)


def extra_columns(cfg: dict, columns) -> list[str]:
    """Stage-1 pair features carried into the prediction files for stage 2 (stack.pair_features)
    and emb_cos (the reranker's pair selection), when the feature parts have them."""
    want = list(cfg.get("stack", {}).get("pair_features", [])) + ["emb_cos"]
    return [c for c in dict.fromkeys(want) if c in columns]


def load_boosters(cfg: dict) -> list[lgb.Booster]:
    return [lgb.Booster(model_file=str(p)) for p in sorted(lgbm_dir(cfg).glob("fold*.txt"))]


def predict_split(cfg: dict, split: str, role: int | None, boosters, names) -> pd.DataFrame:
    """Score a split part by part (s1, t, p and label when available), keeping memory flat."""
    import pyarrow.parquet as pq

    frames = []
    extras = extra_columns(cfg, pq.read_schema(feature_parts(cfg, split)[0]).names)
    for p in feature_parts(cfg, split):
        cols = list(dict.fromkeys(names + extras + ["s1", "t"] + (["label", "role"] if split == "train" else [])))
        df = pd.read_parquet(p, columns=cols)
        if role is not None:
            df = df[df["role"] == role]
        out = pd.DataFrame({"s1": df["s1"].to_numpy(), "t": df["t"].to_numpy(),
                            "p": predict(boosters, df[names].to_numpy(np.float32), n_threads(cfg))})
        if "label" in df:
            out["label"] = df["label"].to_numpy()
        for c in extras:
            out[c] = df[c].to_numpy(np.float32)
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


def run_single(cfg: dict) -> None:
    """lgbm.folds = 1: one model fit on 4/5 of the training-sample S1 groups and early-stopped
    on the other fifth, i.e. exactly the quick protocol of src/experiment.py (its cached fit
    and scores are reused when the features and settings match; otherwise it is fitted here,
    deterministically). With ~2,000+ trees, scoring the 100M+ test pairs with five fold
    models would take hours, so the final model is this single fit.

    Writes the same files as the 5-fold path. train_oof.parquet then holds in-sample scores
    for 4/5 of the training sample (out-of-sample only for the early-stopping fifth), so
    calibration uses role0.parquet (never trained on) instead.
    """
    from .experiment import cached_fit, cached_scores, model_key

    t_start = time.perf_counter()
    names = feature_names(cfg)
    out = lgbm_dir(cfg)
    pdir = preds_dir(cfg)
    pdir.mkdir(parents=True, exist_ok=True)
    stamp = {"model": model_key(cfg, names), "extras": extra_columns(cfg, names + ["emb_cos"])}
    if (pdir / "stamp.json").exists() and json.loads((pdir / "stamp.json").read_text()) == stamp \
            and (out / "fold0.txt").exists():
        log("stage 1 already fitted and scored; skipped")
        return
    booster, best_it, cache = cached_fit(cfg, names)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("fold*.txt"):
        old.unlink()
    booster.save_model(str(out / "fold0.txt"), num_iteration=best_it if best_it > 0 else None)
    scored = cached_scores(cfg, booster, names, cache, roles=(0, 1, 2, 3))
    role = scored["role"].to_numpy()
    keep = [c for c in scored.columns if c != "role"]
    for r, name in ((1, "train_oof"), (2, "val"), (0, "role0"), (3, "role3")):
        scored.loc[role == r, keep].to_parquet(pdir / f"{name}.parquet", index=False)
    (pdir / "stamp.json").write_text(json.dumps(stamp))
    gain = booster.feature_importance("gain")
    importance = sorted(zip(names, gain.tolist()), key=lambda x: -x[1])
    with open(out / "importance.json", "w", encoding="utf-8") as f:
        json.dump({"features": names, "gain": importance, "best_iterations": [best_it]}, f, indent=1)
    log(f"train stage (single fit, best iteration {best_it}) done in {time.perf_counter() - t_start:.0f}s; "
        "top features: " + ", ".join(n for n, _ in importance[:10]))


def run(cfg: dict) -> None:
    """Train stage: fit the fold models, save them, OOF and validation predictions, importances."""
    if cfg["lgbm"]["folds"] == 1:
        return run_single(cfg)
    t_start = time.perf_counter()
    names = feature_names(cfg)
    X, y, s1, t = load_matrix(cfg, "train", 1, names)
    log(f"train: {len(y):,} pairs, {int(y.sum()):,} positives, {len(names)} features, "
        f"matrix {X.nbytes / 2**30:.1f} GB")
    boosters, oof = fit_folds(cfg, X, y, s1, names)
    out = lgbm_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("fold*.txt"):
        old.unlink()
    for i, b in enumerate(boosters):
        b.save_model(str(out / f"fold{i}.txt"), num_iteration=b.best_iteration)
    pdir = preds_dir(cfg)
    pdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"s1": s1, "t": t, "label": y, "p": oof}).to_parquet(pdir / "train_oof.parquet", index=False)
    del X
    predict_split(cfg, "train", 2, boosters, names).to_parquet(pdir / "val.parquet", index=False)
    # Remaining train S1 (role 0, blocked only to complete every target's competitor set):
    # out-of-sample for all fold models, used by the assignment step.
    other = predict_split(cfg, "train", 0, boosters, names)
    if len(other):
        other.to_parquet(pdir / "role0.parquet", index=False)
    gain = np.mean([b.feature_importance("gain") for b in boosters], axis=0)
    importance = sorted(zip(names, gain.tolist()), key=lambda x: -x[1])
    with open(out / "importance.json", "w", encoding="utf-8") as f:
        json.dump({"features": names, "gain": importance,
                   "best_iterations": [b.best_iteration for b in boosters]}, f, indent=1)
    log(f"train stage done in {time.perf_counter() - t_start:.0f}s; top features: "
        + ", ".join(n for n, _ in importance[:10]))


def score_test(cfg: dict) -> pd.DataFrame:
    """Stage-1 p (and the carried pair features) of every test candidate pair, sorted by
    (s1, t); cached in preds/test_stage1.parquet next to the model it came from."""
    path = preds_dir(cfg) / "test_stage1.parquet"
    model = lgbm_dir(cfg) / "fold0.txt"
    stamp = path.with_suffix(".json")
    ident = {"model_mtime": model.stat().st_mtime_ns,
             "parts": [(p.name, p.stat().st_mtime_ns) for p in feature_parts(cfg, "test")]}
    if path.exists() and stamp.exists() and json.loads(stamp.read_text()) == ident:
        return pd.read_parquet(path)
    t0 = time.perf_counter()
    boosters = load_boosters(cfg)
    with open(lgbm_dir(cfg) / "importance.json", encoding="utf-8") as f:
        names = json.load(f)["features"]
    scored = predict_split(cfg, "test", None, boosters, names)
    scored = scored.iloc[np.lexsort((scored["t"].to_numpy(), scored["s1"].to_numpy()))].reset_index(drop=True)
    scored = scored.astype({"s1": np.int32, "t": np.int32, "p": np.float32})
    preds_dir(cfg).mkdir(parents=True, exist_ok=True)
    scored.to_parquet(path.with_name(path.name + ".partial"), index=False)
    path.with_name(path.name + ".partial").replace(path)
    stamp.write_text(json.dumps(ident))
    log(f"stage 1 scored {len(scored):,} test pairs in {time.perf_counter() - t0:.0f}s")
    return scored
