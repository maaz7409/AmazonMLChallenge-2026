"""Stage 0: learned candidate pruner.

The blockers retrieve a wide pool per S1 record (`blocking.k`: B4@30 + B2w@20 + B1@5,
~48 targets). A light LightGBM scores every pool pair from cheap, country-agnostic
evidence (blocker scores and ranks, embedding cosine and its margin over the target's best
other S1, name/address string similarity, IDF-weighted token overlap, house number) and
their ranks inside the S1's pool; each S1 keeps its `prune.k` best pairs (those below
`prune.min_p` are dropped beyond the first `prune.k_min`). That kept set is the candidate
set: candidate_pairs.tsv, and the only pairs the matching models ever score.

Why: with fixed blocker cutoffs of the same budget (B4@10 + B2w@5 + B1@3, ~14 per S1) about
1.4 % of true pairs never reach the model (validation pair recall 0.9857, F0.5 ceiling 0.9962),
while the ~48-pair pool holds 99.6 % of them. Ranking the pool with a model keeps at most 14
candidates per S1 (11.5 on average) at validation pair recall 0.9953 (F0.5 ceiling 0.9986).

Training data: the pool pairs of the embedder's S1 sample (role 3), never used by stage 1,
stage 2 or validation, so the pruner's probability (`prune_p`) is out-of-sample wherever it
is used downstream (it is also a stage-1 feature).

Steps (python -m src.run_pipeline --stage prune --step <step> [--split train|test]):
  fit     features of the role-3 pool pairs, fit, save models/prune/prune.txt
  select  score every pool pair of --split, keep the best per S1, write the candidate parts
          to artifacts/candidates/<split>_pruned/ (plus the recall table on train)
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
from rapidfuzz import fuzz

from .blocking import candidates_dir, coverage
from .features import RF_WORKERS, TokenIndex, _cpdist, _eq_state, load_embeddings
from .io import KEY_BASE, as_arrow, s1_roles, truth_pairs
from .normalize.normalize import load_normalized
from .utils import fork_map, log, n_procs, n_threads

POOL_BLOCKERS = ["b4", "b2w", "b1"]
GROUP_COLS = ["emb_cos", "emb_margin_other", "name_tset", "name_ratio", "addr_tset", "name_jacc",
              "addr_jacc", "b2w_score", "b1_score"]
COLUMNS = ["key", "country", "name_ascii", "name_core", "address_core", "house_no"]


def prune_dir(cfg: dict):
    return cfg["paths"]["models_dir"] / "prune"


def pruned_dir(cfg: dict, split: str):
    """Candidate parts of the pruned (final) candidate set of a split."""
    return cfg["paths"]["artifacts_dir"] / "candidates" / f"{split}_pruned"


class PruneContext:
    """Record tables, token indexes and embeddings of one split, built once per process."""

    def __init__(self, cfg: dict, split: str):
        s1 = load_normalized(cfg, split, "s1", columns=COLUMNS)
        tg = load_normalized(cfg, split, "targets", columns=COLUMNS)
        self.cfg, self.split = cfg, split
        self.S = {c: as_arrow(s1[c]) for c in ("name_ascii", "name_core", "address_core", "house_no")}
        self.T = {c: as_arrow(tg[c]) for c in ("name_ascii", "name_core", "address_core", "house_no")}
        self.tok_name = TokenIndex(s1["name_core"], tg["name_core"])
        self.tok_addr = TokenIndex(s1["address_core"], tg["address_core"])
        self.t_is_s2 = (tg["key"].to_numpy() // KEY_BASE == 2).astype(np.float32)
        self.emb = load_embeddings(cfg, split)
        self.s1_keys, self.t_keys = s1["key"].to_numpy(), tg["key"].to_numpy()
        self.n_targets = len(tg)

    def compute(self, cand: pd.DataFrame) -> dict[str, np.ndarray]:
        """Pruner features of pool pairs (rows sorted by s1, complete S1 groups)."""
        s, t = cand["s1"].to_numpy(), cand["t"].to_numpy()
        f = {}
        for b in POOL_BLOCKERS:
            if f"{b}_rank" in cand:
                rank = cand[f"{b}_rank"].to_numpy().astype(np.float32)
                f[f"{b}_score"] = cand[f"{b}_score"].to_numpy(np.float32)
                f[f"{b}_rank"] = np.where(rank > 0, rank, np.nan).astype(np.float32)
            else:
                f[f"{b}_score"] = f[f"{b}_rank"] = np.full(len(s), np.nan, np.float32)
        if self.emb is not None:
            cos = np.empty(len(s), np.float32)
            for a in range(0, len(s), 262_144):
                b = a + 262_144
                cos[a:b] = np.einsum("ij,ij->i", np.asarray(self.emb["s1"][s[a:b]], np.float32),
                                     np.asarray(self.emb["t"][t[a:b]], np.float32))
            f["emb_cos"] = cos
            if "rev_best" in self.emb:
                is_best = self.emb["rev_best_s1"][t] == s
                other = np.where(is_best, self.emb["rev_second"][t], self.emb["rev_best"][t])
                f["emb_margin_other"] = (cos - other).astype(np.float32)
            else:
                f["emb_margin_other"] = np.full(len(s), np.nan, np.float32)
        else:
            f["emb_cos"] = f["emb_margin_other"] = np.full(len(s), np.nan, np.float32)

        ss, tt = pa.array(s), pa.array(t)

        def col(side, name, rows):
            return side[name].take(rows).to_numpy(zero_copy_only=False)

        a_nc, b_nc = col(self.S, "name_core", ss), col(self.T, "name_core", tt)
        f["name_tset"] = _cpdist(a_nc, b_nc, fuzz.token_set_ratio)
        del a_nc, b_nc
        f["name_ratio"] = _cpdist(col(self.S, "name_ascii", ss), col(self.T, "name_ascii", tt), fuzz.ratio)
        a_ad, b_ad = col(self.S, "address_core", ss), col(self.T, "address_core", tt)
        empty = (a_ad == "") | (b_ad == "")
        v = _cpdist(a_ad, b_ad, fuzz.token_set_ratio)
        v[empty] = np.nan
        f["addr_tset"] = v
        f["t_addr_empty"] = (b_ad == "").astype(np.float32)
        del a_ad, b_ad
        f["house_eq"] = _eq_state(col(self.S, "house_no", ss), col(self.T, "house_no", tt))
        f["name_jacc"] = self.tok_name.jaccard(s, t)
        aj = self.tok_addr.jaccard(s, t)
        aj[empty] = np.nan
        f["addr_jacc"] = aj
        f["t_is_s2"] = self.t_is_s2[t]

        # position inside the S1's pool: count, and rank / gap to the best per signal
        starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
        sizes = np.diff(np.r_[starts, len(s)])
        gid = np.repeat(np.arange(len(starts)), sizes)
        f["n_pool"] = sizes[gid].astype(np.float32)
        for name in GROUP_COLS:
            x = f[name]
            missing = np.isnan(x)
            v = np.where(missing, -1e3, x).astype(np.float32)
            order = np.lexsort((-v, gid))
            rank = np.empty(len(v), np.float32)
            rank[order] = np.arange(len(v)) - starts[gid[order]] + 1
            best = np.maximum.reduceat(v, starts)[gid]
            f[f"{name}_prank"] = np.where(missing, np.nan, rank)
            f[f"{name}_pgap"] = np.where(missing, np.nan, best - v)
        return f


def pool_parts(cfg: dict, split: str) -> list:
    return sorted(candidates_dir(cfg, split).glob("part-*.parquet"))


def _labels(ctx: PruneContext, s: np.ndarray, t: np.ndarray) -> np.ndarray:
    """1 for ground-truth pairs (train only; ctx.true_codes set by the caller)."""
    true_codes = ctx.true_codes
    codes = s.astype(np.int64) * ctx.n_targets + t
    pos = np.minimum(np.searchsorted(true_codes, codes), len(true_codes) - 1)
    return (true_codes[pos] == codes).astype(np.int8)


def _fit_part(ctx: PruneContext, path) -> str | None:
    """Worker: features of one pool part's role-3 pairs, written next to the model."""
    roles = s1_roles(ctx.cfg, ctx.s1_keys)
    cand = pd.read_parquet(path)
    cand = cand[roles[cand["s1"].to_numpy()] == ctx.cfg["prune"].get("train_role", 3)].reset_index(drop=True)
    if not len(cand):
        return None
    f = ctx.compute(cand)
    out = pd.DataFrame({"s1": cand["s1"].to_numpy(), **f})
    out["label"] = _labels(ctx, cand["s1"].to_numpy(), cand["t"].to_numpy())
    dest = prune_dir(ctx.cfg) / "fit_data" / path.name
    out.to_parquet(dest, index=False)
    return str(dest)


def fit(cfg: dict) -> None:
    """Fit the pruner on the role-3 pool pairs (cached: skipped when the model exists)."""
    import lightgbm as lgb
    from sklearn.model_selection import GroupKFold

    model_path = prune_dir(cfg) / "prune.txt"
    if model_path.exists():
        log(f"pruner already fitted ({model_path}); skipped")
        return
    t0 = time.perf_counter()
    (prune_dir(cfg) / "fit_data").mkdir(parents=True, exist_ok=True)
    ctx = PruneContext(cfg, "train")
    ts, tt = truth_pairs(cfg, ctx.s1_keys, ctx.t_keys)
    ctx.true_codes = np.sort(ts.astype(np.int64) * ctx.n_targets + tt)
    procs = n_procs(cfg)
    RF_WORKERS["n"] = max(1, n_threads(cfg) // procs)
    paths = [p for p in fork_map(_fit_part, ctx, pool_parts(cfg, "train"), procs) if p]
    del ctx
    df = pd.concat([pd.read_parquet(p) for p in sorted(paths)], ignore_index=True)
    names = [c for c in df.columns if c not in ("s1", "label")]
    y, groups = df["label"].to_numpy(), df["s1"].to_numpy()
    X = df[names].to_numpy(np.float32)
    del df
    log(f"pruner training: {len(y):,} pool pairs, {int(y.sum()):,} positive, {len(names)} features "
        f"(features {time.perf_counter() - t0:.0f}s)")
    pc = cfg["prune"]
    params = dict(pc["params"])
    params.update(objective="binary", metric=["binary_logloss", "auc"], seed=cfg["seed"], deterministic=True,
                  force_row_wise=True, verbose=-1, num_threads=n_threads(cfg))
    tr, es = next(GroupKFold(n_splits=5).split(np.zeros((len(y), 1)), y, groups))
    full = lgb.Dataset(X, y, feature_name=names, free_raw_data=True).construct()
    booster = lgb.train(params, full.subset(np.sort(tr)), num_boost_round=pc["max_rounds"],
                        valid_sets=[full.subset(np.sort(es))], valid_names=["es"],
                        callbacks=[lgb.early_stopping(pc["early_stopping"], first_metric_only=True, verbose=False),
                                   lgb.log_evaluation(100)])
    prune_dir(cfg).mkdir(parents=True, exist_ok=True)
    booster.save_model(str(model_path.with_suffix(".partial")), num_iteration=booster.best_iteration)
    model_path.with_suffix(".partial").replace(model_path)
    gain = sorted(zip(names, booster.feature_importance("gain").tolist()), key=lambda x: -x[1])
    (prune_dir(cfg) / "importance.json").write_text(json.dumps({"features": names, "gain": gain,
                                                                 "best_iteration": booster.best_iteration}, indent=1))
    for p in paths:
        Path(p).unlink(missing_ok=True)
    log(f"pruner fitted in {time.perf_counter() - t0:.0f}s: best iteration {booster.best_iteration}, "
        f"auc {booster.best_score['es']['auc']:.5f}; top: " + ", ".join(n for n, _ in gain[:8]))


def select_mask(s: np.ndarray, p: np.ndarray, pc: dict, b4_rank=None) -> tuple[np.ndarray, np.ndarray]:
    """(keep mask, rank inside the S1 by p, 0 = best) for rows sorted by s1: the best
    `k` by p (ranks k_min..k-1 only with p >= min_p), plus the S1's `always_b4` nearest
    embedding neighbours (a safety net where the pruner, fitted on the training countries,
    transfers badly; they are almost always among the best k anyway)."""
    starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
    gid = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(s)]))
    order = np.lexsort((-p, gid))
    rank = np.empty(len(p), np.int32)
    rank[order] = np.arange(len(p)) - starts[gid[order]]
    keep = (rank < pc["k"]) & ((p >= pc["min_p"]) | (rank < pc["k_min"]))
    if b4_rank is not None and pc.get("always_b4"):
        keep |= (b4_rank > 0) & (b4_rank <= pc["always_b4"])
    return keep, rank


def _select_part(ctx: PruneContext, path) -> dict:
    """Worker: score one pool part, keep the best pairs per S1, write the pruned part."""
    import lightgbm as lgb

    pc = ctx.cfg["prune"]
    booster = lgb.Booster(model_file=str(prune_dir(ctx.cfg) / "prune.txt"))
    names = booster.feature_name()
    cand = pd.read_parquet(path)
    f = ctx.compute(cand)
    X = np.column_stack([f[n] for n in names]).astype(np.float32)
    p = booster.predict(X, num_threads=ctx.part_threads).astype(np.float32)
    del X, f
    s = cand["s1"].to_numpy()
    b4r = cand["b4_rank"].to_numpy() if "b4_rank" in cand else None
    keep, rank = select_mask(s, p, pc, b4r)
    out = cand.assign(prune_p=p)
    out_path = pruned_dir(ctx.cfg, ctx.split) / path.name
    tmp = out_path.with_name(out_path.name + ".partial")
    out[keep].reset_index(drop=True).to_parquet(tmp, index=False)
    tmp.replace(out_path)
    res = {"part": path.name, "pool": len(cand), "kept": int(keep.sum())}
    if ctx.split == "train":  # rank of every pool pair, for the recall-vs-K table on validation
        roles = s1_roles(ctx.cfg, ctx.s1_keys)
        v = roles[s] == 2
        res["val"] = {"s1": s[v], "t": cand["t"].to_numpy()[v], "rank": rank[v], "p": p[v], "keep": keep[v]}
    return res


def select(cfg: dict, split: str) -> None:
    """Score and prune every pool part of a split (parts already pruned are skipped)."""
    t0 = time.perf_counter()
    out_dir = pruned_dir(cfg, split)
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = pool_parts(cfg, split)
    stamp = {"model_mtime": (prune_dir(cfg) / "prune.txt").stat().st_mtime_ns,
             "prune": cfg["prune"], "pool": [(p.name, p.stat().st_mtime_ns) for p in parts]}
    stamp_path = out_dir / "stamp.json"
    if stamp_path.exists() and json.loads(stamp_path.read_text()) == json.loads(json.dumps(stamp)):
        log(f"pruned candidates [{split}] up to date; skipped")
        return
    for old in list(out_dir.glob("part-*.parquet")) + list(out_dir.glob("*.partial")):
        old.unlink()
    stamp_path.unlink(missing_ok=True)
    ctx = PruneContext(cfg, split)
    procs = n_procs(cfg)
    ctx.part_threads = max(1, n_threads(cfg) // procs)
    RF_WORKERS["n"] = ctx.part_threads
    results = fork_map(_select_part, ctx, parts, procs)
    pool, kept = sum(r["pool"] for r in results), sum(r["kept"] for r in results)
    n_s1 = len(ctx.s1_keys)
    report = {"split": split, "pool_pairs": pool, "kept_pairs": kept,
              "mean_pool_per_s1": round(pool / n_s1, 2), "mean_kept_per_s1": round(kept / n_s1, 2),
              "settings": cfg["prune"], "seconds": round(time.perf_counter() - t0, 1)}
    if split == "train":
        report["recall"] = recall_table(cfg, ctx, [r["val"] for r in results])
    (out_dir / "prune_report.json").write_text(json.dumps(report, indent=1, default=str))
    stamp_path.write_text(json.dumps(stamp))
    log(f"pruned [{split}]: {pool:,} pool pairs -> {kept:,} kept ({report['mean_kept_per_s1']} per S1) "
        f"in {report['seconds']}s")
    if "recall" in report:
        print(json.dumps(report["recall"], indent=1))


def recall_table(cfg: dict, ctx: PruneContext, vals: list[dict]) -> dict:
    """Validation recall of the pool and of the pruned set at several K, overall and per country."""
    s = np.concatenate([v["s1"] for v in vals])
    t = np.concatenate([v["t"] for v in vals])
    rank = np.concatenate([v["rank"] for v in vals])
    ts, tt = truth_pairs(cfg, ctx.s1_keys, ctx.t_keys)
    roles = s1_roles(cfg, ctx.s1_keys)
    val = np.flatnonzero(roles == 2)
    n_s1, n_t = len(ctx.s1_keys), ctx.n_targets
    pc = cfg["prune"]
    out = {"pool": coverage(s, t, ts, tt, val, n_s1, n_t)}
    for k in sorted({8, 10, 12, 14, 16, 20, pc["k"]}):
        m = rank < k
        out[f"top{k}"] = coverage(s[m], t[m], ts, tt, val, n_s1, n_t)
    m = np.concatenate([v["keep"] for v in vals])
    out["final"] = coverage(s[m], t[m], ts, tt, val, n_s1, n_t)
    country = load_normalized(cfg, "train", "s1", columns=["country"])["country"].to_numpy(dtype=object)
    out["final_per_country"] = {c: coverage(s[m], t[m], ts, tt, val[country[val] == c], n_s1, n_t)
                                for c in pd.unique(country[val])}
    return out


def run(cfg: dict, step: str, split: str) -> None:
    """Dispatch one pruner step."""
    if step == "fit":
        fit(cfg)
    elif step in ("select", None):
        select(cfg, split)
    else:
        raise SystemExit(f"unknown prune step '{step}'")
