"""Stage 2: a pair model over the
stage-1 probabilities of the whole candidate graph.

Stage 1 (LightGBM, src/train.py) scores each (S1, target) pair from that pair's text
alone. Stage 2 re-scores every pair from its stage-1 probability p and its neighbourhood:

* its S1's candidate list: best and second p, rank, p relative to the best, the number
  of confident candidates and the sum of p (an expected match count), available to every
  pair of the S1 (they also decide whether the S1 is a singleton);
* its target's competition: the best p any OTHER S1 has for this target, the margin to
  it, whether this S1 is the target's argmax (in the labels, a target is in at most one S1
  list) and how many other S1 claim the target;
* siblings: the target's name/address similarity (and, with blocker B4, embedding
  cosine) to the S1's two other top candidates, and their p. One business usually appears
  several times across S2 and S3, so a confident match vouches for its near-duplicates.

Training data: the pairs of a seeded sample of role-0 train S1 records. Stage 1 never saw
them, so their p is out-of-sample exactly as on validation and test (all three are scored
by the same stage-1 model). Early stopping uses a held-out fifth of those S1 records; the
decision rule is then tuned on validation stage-2 probabilities. The candidate set is
unchanged. Country is not a feature.
"""

import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
from rapidfuzz import fuzz, process
from sklearn.model_selection import GroupKFold

from .decide import groups
from .utils import log, n_threads

STACK_FEATURES = [
    "p", "g_best_p", "g_second_p", "p_rank", "p_rel_best", "p_gap_best", "g_n_cands", "g_sum_p", "g_n_hi",
    "t_other_best_p", "p_margin_t", "t_is_argmax", "t_n_other_s1", "t_n_other_hi",
    "sib1_p", "sib1_name_sim", "sib1_addr_sim", "sib1_same_src", "sib1_emb_cos",
    "sib2_p", "sib2_name_sim", "sib2_addr_sim", "sib2_same_src", "sib2_emb_cos",
    "sib_support", "sib_emb_support",
]
# Reranker features (config stack.rerank): the mmBERT reranker's probability for the pairs it re-scored
# (NaN elsewhere), its gap to the S1's best, the S1's confident count, and target competition.
RERANK_FEATURES = ["rr_p", "rr_gap_best", "rr_n_hi", "rr_t_other_best", "rr_margin_t",
                   "rr_t_is_argmax", "rr_t_n_other_hi"]
# Prefixes of reranker feature blocks. "rr" = the reranker of the run (default), "rw" = the
# raw-text reranker (c) when stage 2 reads it next to another one, "rx" = the second raw
# reranker (d). The final variant rawpl2 reads rr (reranker (b)), rw and rx.
RERANK_PREFIXES = ("rr", "rw", "rx")


def rerank_blocks(cfg: dict) -> list[tuple[str, str]]:
    """(feature prefix, score-file tag) of each reranker stage 2 reads: by default one block
    "rr" from scores_<split><run tag>.parquet; `stack.rerank_blocks` lists several,
    e.g. [["rr", ""], ["rw", "_raw"]] = the main (normalized-text) and the raw-text reranker."""
    blocks = [tuple(b) for b in cfg["stack"].get("rerank_blocks") or [("rr", run_tag(cfg))]]
    assert all(pre in RERANK_PREFIXES for pre, _ in blocks), blocks
    return blocks


def block_features(prefix: str) -> list[str]:
    """RERANK_FEATURES of one reranker block ("rr" gives RERANK_FEATURES itself)."""
    return [prefix + name[2:] for name in RERANK_FEATURES]


def feature_list(cfg: dict) -> list[str]:
    """Stage-2 inputs: STACK_FEATURES, the stage-1 pair features of stack.pair_features (so
    stage 2 can re-weigh the raw evidence against the graph context), RERANK_FEATURES of each
    reranker block when stack.rerank is on, and the columns of stack.extra_pairs (per-pair
    evidence of another pipeline, see attach_extra_pairs)."""
    rerank = [f for pre, _ in rerank_blocks(cfg) for f in block_features(pre)] if cfg["stack"].get("rerank") else []
    extra = list((cfg["stack"].get("extra_pairs") or {}).get("columns", []))
    return STACK_FEATURES + list(cfg["stack"].get("pair_features", [])) + rerank + extra


def attach_extra_pairs(cfg: dict, scored: pd.DataFrame, split: str) -> pd.DataFrame:
    """Optional (`stack.extra_pairs`: {train: file, test: file, columns: [...]}): per-pair features
    of ANOTHER pipeline (e.g. its stage-1 and reranker probabilities, s1/t = raw-file row
    positions, so the indices match) joined onto the scored pairs; NaN where that pipeline did
    not have the pair. Stage 2 learns on the training countries when to trust which pipeline.
    Paths are relative to the artifacts dir. Only the "x" variants use it (raw.extra_pairs);
    the files come from a pipeline that is not part of this repository."""
    x = cfg["stack"].get("extra_pairs")
    if not x:
        return scored
    path = Path(x[split])
    if not path.is_absolute():
        path = cfg["paths"]["artifacts_dir"] / path
    ext = pd.read_parquet(path, columns=["s1", "t"] + list(x["columns"]))
    n_t = int(max(scored["t"].max(), ext["t"].max())) + 1
    codes = scored["s1"].to_numpy().astype(np.int64) * n_t + scored["t"].to_numpy()
    ec = ext["s1"].to_numpy().astype(np.int64) * n_t + ext["t"].to_numpy()
    order = np.argsort(ec)
    ec = ec[order]
    pos = np.minimum(np.searchsorted(ec, codes), len(ec) - 1)
    hit = ec[pos] == codes
    for c in x["columns"]:
        v = ext[c].to_numpy(np.float32)[order]
        out = np.full(len(scored), np.nan, np.float32)
        out[hit] = v[pos[hit]]
        scored = scored.assign(**{c: out})
    log(f"extra pair features {list(x['columns'])} attached to {int(hit.sum()):,} of {len(scored):,} {split} pairs "
        f"from {path.name}")
    return scored


def target_competition_p(t: np.ndarray, p: np.ndarray) -> dict[str, np.ndarray]:
    """Per pair: the best stage-1 p of ANOTHER S1 for the same target, whether this pair is
    the target's argmax, and how many other S1 claim the target (in total / with p > 0.5).

    Must be given every scored pair of the universe (all S1 roles on train), or the
    competition is incomplete.
    """
    order = np.lexsort((-p, t))
    ts, ps = t[order], p[order]
    first = np.r_[True, ts[1:] != ts[:-1]]
    starts = np.flatnonzero(first)
    gid = np.cumsum(first) - 1
    size = np.diff(np.r_[starts, len(ts)])
    best = ps[starts]
    second = np.where(size > 1, ps[np.minimum(starts + 1, len(ps) - 1)], 0.0)
    hi = (ps > 0.5).astype(np.int32)
    n_hi = np.add.reduceat(hi, starts)
    sorted_values = {
        "t_other_best_p": np.where(first, second[gid], best[gid]),  # 0 when no other S1
        "t_is_argmax": first,
        "t_n_other_s1": size[gid] - 1,
        "t_n_other_hi": n_hi[gid] - hi,
    }
    out = {}
    for name, v in sorted_values.items():
        arr = np.empty(len(p), np.float32)
        arr[order] = v
        out[name] = arr
    return out


def competition(scored: pd.DataFrame) -> dict[str, np.ndarray]:
    """target_competition_p over every scored pair; with a reranker column (rr_p, and rw_p
    for a second block), also the best reranker probability of another S1 for the same
    target, over the re-scored pairs (NaN for pairs the reranker did not score)."""
    t = scored["t"].to_numpy()
    comp = target_competition_p(t, scored["p"].to_numpy())
    rr_cols = {f"{pre}_p" for pre in RERANK_PREFIXES}
    for c in scored.columns:  # stage-1 pair features carried in the prediction files
        if c not in ("s1", "t", "p", "label", "role", "p1") and c not in rr_cols:
            comp[c] = scored[c].to_numpy(np.float32)
    for pre in RERANK_PREFIXES:
        if f"{pre}_p" not in scored:
            continue
        rr = scored[f"{pre}_p"].to_numpy()
        ok = np.flatnonzero(np.isfinite(rr))
        other = np.full(len(rr), np.nan, np.float32)
        is_arg = np.full(len(rr), np.nan, np.float32)
        n_hi = np.full(len(rr), np.nan, np.float32)
        c = target_competition_p(t[ok], rr[ok])
        other[ok], is_arg[ok], n_hi[ok] = c["t_other_best_p"], c["t_is_argmax"], c["t_n_other_hi"]
        comp[f"{pre}_p"] = rr.astype(np.float32)
        comp[f"{pre}_t_other_best"] = other
        comp[f"{pre}_t_is_argmax"] = is_arg
        comp[f"{pre}_t_n_other_hi"] = n_hi
    return comp


def run_tag(cfg: dict) -> str:
    """Suffix of every stage-2 artifact of this run: "" for the main run, "_pl" for the
    optional pseudo-label round (its own reranker scores, stage 2, rule and outputs)."""
    return cfg["stack"].get("rerank_tag", "")


def attach_rerank(cfg: dict, scored: pd.DataFrame, split: str) -> pd.DataFrame:
    """With stack.rerank on: add the cached probabilities of each reranker block (rr_p, and
    rw_p for a second block; NaN for pairs it did not score) to scored pairs sorted by (s1, t)."""
    if not cfg["stack"].get("rerank"):
        return scored
    for pre, tag in rerank_blocks(cfg):
        rr = pd.read_parquet(cfg["paths"]["artifacts_dir"] / "rerank" / f"scores_{split}{tag}.parquet")
        n_t = int(max(scored["t"].max(), rr["t"].max())) + 1
        codes = scored["s1"].to_numpy().astype(np.int64) * n_t + scored["t"].to_numpy()
        rr_codes = rr["s1"].to_numpy().astype(np.int64) * n_t + rr["t"].to_numpy()
        order = np.argsort(rr_codes)
        rr_codes, rr_p = rr_codes[order], rr["rr_p"].to_numpy()[order]
        pos = np.minimum(np.searchsorted(rr_codes, codes), len(rr_codes) - 1)
        hit = rr_codes[pos] == codes
        out = np.full(len(scored), np.nan, np.float32)
        out[hit] = rr_p[pos[hit]]
        log(f"reranker scores{tag} ({pre}_p) attached to {int(hit.sum()):,} of {len(scored):,} {split} pairs")
        scored = scored.assign(**{f"{pre}_p": out})
    return attach_extra_pairs(cfg, scored, split)


def _sim(a_idx: np.ndarray, b_idx: np.ndarray, col: pa.Array, scorer, chunk: int = 1_000_000) -> np.ndarray:
    """token-set similarity (0-100) of target strings a vs b; NaN when b is absent (-1) or
    either string is empty. Chunked so the Python string lists stay small."""
    out = np.full(len(a_idx), np.nan, np.float32)
    ok = np.flatnonzero(b_idx >= 0)
    for i in range(0, len(ok), chunk):
        rows = ok[i:i + chunk]
        a = col.take(pa.array(a_idx[rows])).to_pylist()
        b = col.take(pa.array(b_idx[rows])).to_pylist()
        v = process.cpdist(a, b, scorer=scorer, processor=None, workers=-1, dtype=np.float32)
        empty = np.fromiter((not x or not y for x, y in zip(a, b)), bool, len(a))
        v[empty] = np.nan
        out[rows] = v
    return out


def _emb_cos(a_idx: np.ndarray, b_idx: np.ndarray, emb, chunk: int = 262_144) -> np.ndarray:
    """Cosine of target embeddings a vs b (L2-normalized rows, so a dot product); NaN when
    b is absent (-1) or no embeddings are configured. Chunked (a gather of 8M 256-d vectors
    would take 8 GB)."""
    out = np.full(len(a_idx), np.nan, np.float32)
    if emb is None:
        return out
    ok = np.flatnonzero(b_idx >= 0)
    for i in range(0, len(ok), chunk):
        rows = ok[i:i + chunk]
        out[rows] = np.einsum("ij,ij->i", np.asarray(emb[a_idx[rows]], np.float32),
                              np.asarray(emb[b_idx[rows]], np.float32))
    return out


def graph_features(s1: np.ndarray, t: np.ndarray, p: np.ndarray, comp: dict[str, np.ndarray],
                   tgt: dict, names: list[str]) -> np.ndarray:
    """Stage-2 features of whole S1 groups: rows sorted by s1; `comp` is competition() of the
    full universe restricted to these rows; `tgt` is target_data() of the universe.
    Returns a float32 matrix with the columns `names` (feature_list of the config)."""
    starts, gid = groups(s1)
    size = np.diff(np.r_[starts, len(s1)])
    order = np.lexsort((-p, gid))  # by group, p descending; group boundaries stay `starts`
    ps, ts_ = p[order], t[order]
    rank = np.empty(len(p), np.int32)
    rank[order] = np.arange(len(p)) - starts[gid[order]]  # 0 = the S1's best candidate
    last = len(p) - 1

    def kth(values, k, fill):  # per group: k-th best value (0-based), `fill` if absent
        return np.where(size > k, values[np.minimum(starts + k, last)], fill)

    best, second, third = kth(ps, 0, 0.0), kth(ps, 1, 0.0), kth(ps, 2, 0.0)
    top = [kth(ts_, k, -1) for k in range(3)]
    g = gid
    f = {"p": p, "g_best_p": best[g], "g_second_p": second[g], "p_rank": rank + 1.0,
         "p_rel_best": p / np.maximum(best[g], 1e-6), "p_gap_best": best[g] - p,
         "g_n_cands": size[g], "g_sum_p": np.add.reduceat(p, starts)[g],
         "g_n_hi": np.add.reduceat((p > 0.5).astype(np.int32), starts)[g]}
    f.update(comp)
    f["p_margin_t"] = p - comp["t_other_best_p"]
    # siblings: the S1's two best candidates other than this pair
    sib = {1: (np.where(rank == 0, top[1][g], top[0][g]), np.where(rank == 0, second[g], best[g])),
           2: (np.where(rank <= 1, top[2][g], top[1][g]), np.where(rank <= 1, third[g], second[g]))}
    support = np.zeros(len(p), np.float32)
    emb_support = np.zeros(len(p), np.float32)
    for k, (st, sp) in sib.items():
        name_sim = _sim(t, st, tgt["name"], fuzz.token_set_ratio)
        addr_sim = _sim(t, st, tgt["addr"], fuzz.token_set_ratio)
        cos = _emb_cos(t, st, tgt.get("emb"))
        sp = np.where(st >= 0, sp, 0.0)
        f[f"sib{k}_p"] = np.where(st >= 0, sp, np.nan)
        f[f"sib{k}_name_sim"] = name_sim
        f[f"sib{k}_addr_sim"] = addr_sim
        f[f"sib{k}_same_src"] = np.where(st >= 0, tgt["source"][t] == tgt["source"][np.maximum(st, 0)], np.nan)
        f[f"sib{k}_emb_cos"] = cos
        both = np.nan_to_num(np.minimum(name_sim, addr_sim), nan=0.0) / 100
        support = np.maximum(support, sp * both)
        emb_support = np.maximum(emb_support, sp * np.clip(np.nan_to_num(cos, nan=0.0), 0.0, 1.0))
    f["sib_support"] = support
    f["sib_emb_support"] = emb_support if tgt.get("emb") is not None else np.full(len(p), np.nan, np.float32)
    for pre in RERANK_PREFIXES:  # reranker features (stack.rerank), one block per reranker
        if f"{pre}_p" not in comp or not any(n in names for n in block_features(pre)):
            continue
        rr = comp[f"{pre}_p"]
        best = np.maximum.reduceat(np.where(np.isfinite(rr), rr, -np.inf), starts)
        best = np.where(np.isfinite(best), best, np.nan)[g]
        f[f"{pre}_gap_best"] = best - rr
        f[f"{pre}_n_hi"] = np.add.reduceat((np.nan_to_num(rr, nan=0.0) > 0.5).astype(np.int32), starts)[g]
        f[f"{pre}_margin_t"] = rr - np.nan_to_num(comp[f"{pre}_t_other_best"], nan=0.0)
    X = np.empty((len(p), len(names)), np.float32)
    for j, k in enumerate(names):
        X[:, j] = f[k] if k in f else np.nan  # a configured pair feature the files lack
    return X


def chunk_bounds(s1: np.ndarray, s1_per_chunk: int) -> list[tuple[int, int]]:
    """Row ranges of consecutive whole S1 groups (rows sorted by s1)."""
    starts, _ = groups(s1)
    cuts = np.r_[starts[::s1_per_chunk], len(s1)]
    return list(zip(cuts[:-1], cuts[1:]))


def features_for(scored: pd.DataFrame, rows: np.ndarray, comp: dict, tgt: dict, names: list[str],
                 s1_per_chunk: int = 100_000) -> np.ndarray:
    """graph_features for the selected rows (whole S1 groups, sorted by s1), chunk by chunk,
    into one preallocated float32 matrix (no DataFrame copies: 16M training rows take 1.7 GB)."""
    sub = scored.iloc[rows] if rows is not None else scored
    s1, t, p = sub["s1"].to_numpy(), sub["t"].to_numpy(), sub["p"].to_numpy()
    c = {k: (v[rows] if rows is not None else v) for k, v in comp.items()}
    X = np.empty((len(s1), len(names)), np.float32)
    for a, b in chunk_bounds(s1, s1_per_chunk):
        X[a:b] = graph_features(s1[a:b], t[a:b], p[a:b], {k: v[a:b] for k, v in c.items()}, tgt, names)
    return X


def params(cfg: dict) -> dict:
    p = dict(cfg["stack"]["params"])
    p.update(objective="binary", metric=["binary_logloss", "auc"], seed=cfg["seed"],
             deterministic=True, force_row_wise=True, verbose=-1, num_threads=n_threads(cfg))
    return p


def cache_name(cfg: dict) -> str:
    """File name of a stage-2 model inside its stage-1 fit's cache directory (the name
    hashes everything else stage 2 depends on)."""
    import hashlib
    import json
    ident = {"params": {k: v for k, v in params(cfg).items() if k != "num_threads"},
             "train_s1": cfg["stack"]["train_s1"], "features": feature_list(cfg),
             "rerank_tag": cfg["stack"].get("rerank_tag", ""),
             "complete_competition": bool(cfg["rerank"].get("complete_competition")),
             "rounds": cfg["stack"]["max_rounds"], "early_stopping": cfg["stack"]["early_stopping"]}
    return "stack_" + hashlib.sha1(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:12] + ".txt"


def cached_fit(cfg: dict, cache_dir, X_fn, y_fn, s1_fn) -> lgb.Booster:
    """fit() cached as cache_dir/cache_name(cfg); X_fn/y_fn/s1_fn build the training data
    only when no cached model exists."""
    path = cache_dir / cache_name(cfg) if cache_dir is not None else None
    if path is not None and path.exists():
        log(f"stage 2: cached model {path.name}")
        return lgb.Booster(model_file=str(path))
    booster = fit(cfg, X_fn(), y_fn(), s1_fn())
    if path is not None:
        booster.save_model(str(path) + ".partial", num_iteration=booster.best_iteration)
        Path(str(path) + ".partial").replace(path)
    return booster


def fit(cfg: dict, X: np.ndarray, y: np.ndarray, s1: np.ndarray) -> lgb.Booster:
    """Stage-2 LightGBM, early-stopped on a held-out fifth of the training S1 groups."""
    tr, es = next(GroupKFold(n_splits=5).split(np.zeros((len(y), 1)), y, s1))
    names = feature_list(cfg)
    assert X.shape[1] == len(names), (X.shape, len(names))
    full = lgb.Dataset(X, y, feature_name=names, free_raw_data=True).construct()
    t0 = time.perf_counter()
    booster = lgb.train(params(cfg), full.subset(np.sort(tr)), num_boost_round=cfg["stack"]["max_rounds"],
                        valid_sets=[full.subset(np.sort(es))], valid_names=["es"],
                        callbacks=[lgb.early_stopping(cfg["stack"]["early_stopping"], first_metric_only=True,
                                                      verbose=False)])
    log(f"stage 2: {len(y):,} training pairs, best iteration {booster.best_iteration}, "
        f"auc {booster.best_score['es']['auc']:.5f}, {time.perf_counter() - t0:.0f}s")
    return booster


def train_s1_mask(cfg: dict, roles: np.ndarray, s1_country=None, country=None) -> np.ndarray:
    """Boolean mask over S1 rows: the seeded sample of role-0 S1 records (optionally of one
    country) whose pairs train stage 2."""
    pool = np.flatnonzero(roles == 0)
    if country is not None:
        pool = pool[s1_country[pool] == country]
    rng = np.random.default_rng(cfg["seed"])
    chosen = np.zeros(len(roles), bool)
    chosen[rng.choice(pool, size=min(len(pool), cfg["stack"]["train_s1"]), replace=False)] = True
    return chosen


def train_rows(cfg: dict, scored: pd.DataFrame, roles: np.ndarray, s1_country=None, country=None) -> np.ndarray:
    """Rows of a seeded sample of role-0 S1 records (optionally one country)."""
    return np.flatnonzero(train_s1_mask(cfg, roles, s1_country, country)[scored["s1"].to_numpy()])


def predict_rows(booster, scored, rows, comp, tgt: dict, threads: int = 0) -> np.ndarray:
    """Stage-2 probabilities for the selected rows (whole S1 groups), chunk by chunk."""
    sub_s1 = scored["s1"].to_numpy()[rows]
    names = booster.feature_name()
    out = np.empty(len(rows), np.float32)
    kw = {"num_threads": threads} if threads else {}
    for a, b in chunk_bounds(sub_s1, 100_000):
        X = features_for(scored, rows[a:b], comp, tgt, names)
        out[a:b] = booster.predict(X, num_iteration=booster.best_iteration, **kw)
    return out


# ---- pipeline integration (config stack.enabled) ------------------------------------------

def model_path(cfg: dict):
    return cfg["paths"]["models_dir"] / "stack" / f"stack{run_tag(cfg)}.txt"


def target_data(cfg: dict, split: str) -> dict:
    """A split's targets for stage 2: source number, name_core, address_core and, with a B4
    model configured, the (memory-mapped) target embeddings."""
    from .io import KEY_BASE, as_arrow
    from .normalize.normalize import load_normalized

    tg = load_normalized(cfg, split, "targets", columns=["key", "name_core", "address_core"])
    out = {"source": (tg["key"].to_numpy() // KEY_BASE).astype(np.int8),
           "name": as_arrow(tg["name_core"]), "addr": as_arrow(tg["address_core"]), "emb": None}
    tag = cfg["blocking"].get("b4")
    if tag:
        out["emb"] = np.load(cfg["paths"]["artifacts_dir"] / "embeddings" / f"{split}_targets_{tag}.npy",
                             mmap_mode="r")
    return out


def rescore(booster, scored: pd.DataFrame, tgt: dict, threads: int = 0) -> np.ndarray:
    """Stage-2 p for every pair of a complete universe (rows sorted by s1; an rr_p column,
    when present, supplies the reranker features)."""
    comp = competition(scored)
    return predict_rows(booster, scored, np.arange(len(scored)), comp, tgt, threads)


def load_train_scored(cfg: dict) -> pd.DataFrame:
    """Every scored train pair (all S1 roles, stage-1 p and the carried pair features),
    sorted by (s1, t)."""
    from .train import preds_dir

    pdir = preds_dir(cfg)
    frames = [pd.read_parquet(pdir / f"{name}.parquet") for name in ("train_oof", "val", "role0", "role3")
              if (pdir / f"{name}.parquet").exists()]
    scored = pd.concat(frames, ignore_index=True)
    del frames
    scored = scored.astype({"s1": np.int32, "t": np.int32, "label": np.int8, "p": np.float32})
    return scored.iloc[np.lexsort((scored["t"].to_numpy(), scored["s1"].to_numpy()))].reset_index(drop=True)


def run(cfg: dict) -> None:
    """Train stage 2 on the stage-1 predictions of the train stage and write validation p.

    Reads preds/{train_oof,val,role0,role3}.parquet (every train pair), fits stage 2 on the
    role-0 sample, saves models/stack/stack<tag>.txt and preds/val_stack<tag>.parquet
    (validation pairs, stage-2 p).
    """
    from .io import s1_roles
    from .normalize.normalize import load_normalized
    from .train import preds_dir

    t0 = time.perf_counter()
    pdir = preds_dir(cfg)
    scored = load_train_scored(cfg)
    s1_keys = load_normalized(cfg, "train", "s1", columns=["key"])["key"].to_numpy()
    roles = s1_roles(cfg, s1_keys)
    tgt = target_data(cfg, "train")
    scored = attach_rerank(cfg, scored, "train")
    comp = competition(scored)
    tr = train_rows(cfg, scored, roles)
    cache_dir = None
    if cfg["lgbm"]["folds"] == 1:  # single-fit stage 1: reuse the quick protocol's stage 2 if cached
        from .experiment import model_key
        from .train import feature_names
        cache_dir = cfg["paths"]["artifacts_dir"] / "experiments" / "cache" / model_key(cfg, feature_names(cfg))
    names = feature_list(cfg)
    booster = cached_fit(cfg, cache_dir, lambda: features_for(scored, tr, comp, tgt, names),
                         lambda: scored["label"].to_numpy()[tr], lambda: scored["s1"].to_numpy()[tr])
    path = model_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(path), num_iteration=booster.best_iteration)
    vr = np.flatnonzero(roles[scored["s1"].to_numpy()] == 2)
    p2 = predict_rows(booster, scored, vr, comp, tgt, n_threads(cfg))
    val = scored.iloc[vr][["s1", "t", "label"]].reset_index(drop=True).assign(p=p2)
    val.to_parquet(pdir / f"val_stack{run_tag(cfg)}.parquet", index=False)
    if cfg["stack"].get("full_train_p"):  # stage-2 p of EVERY train pair (decide.assignment
        # needs the complete competition for validation targets) -> preds/train_stack<tag>.parquet
        t1 = time.perf_counter()
        p_all = predict_rows(booster, scored, np.arange(len(scored)), comp, tgt, n_threads(cfg))
        scored[["s1", "t", "label"]].assign(p=p_all).to_parquet(pdir / f"train_stack{run_tag(cfg)}.parquet", index=False)
        log(f"stage 2: p of all {len(scored):,} train pairs in {time.perf_counter() - t1:.0f}s (train_stack{run_tag(cfg)})")
    gain = sorted(zip(booster.feature_name(), booster.feature_importance("gain").tolist()), key=lambda x: -x[1])
    log(f"stage 2 done in {time.perf_counter() - t0:.0f}s; top features: "
        + ", ".join(f"{n} {g:.0f}" for n, g in gain[:10]))
