"""jhu-clsp/mmBERT-base cross-encoder rerankers, stage-2 features.

The cross-encoder reads both records together, "{name_ascii} | {address_norm}" for the S1
record and for the target (in random order during training), and outputs P(match). It
re-scores the pairs that decide the output (`rerank.score_*`): every candidate
with stage-1 p >= `score_low`, each S1's `score_top` best candidates by stage-1 p, and its
`score_top_emb` nearest by embedding cosine. The last group does not depend on stage 1, the
component that transfers worst to an unseen country. Its probability becomes a stage-2
feature (src/stack.py); unscored pairs get NaN.

Leakage: the reranker trains on pairs of role-1 S1 (the stage-1 training sample). Stage 2
trains on role-0 S1 and is tuned on validation (role 2), so every reranker score stage 2
sees (role 0, validation, test) comes from a model that never saw those S1 records.

Training pairs (per S1): every true pair; up to `neg_per_pos` negatives per positive (as many
as the S1's candidate list holds), 75% the hardest (by the pruner's probability with the pruned
candidates, else by stage-1 p) and 25% at random from the rest.

Round 2, pseudo-label round (`--stage pseudo`, reranker (b); it trains on the model's own
confident test predictions, see README.md): after a complete first run, S1 records of
test whose decision is unambiguous (every emitted pair p >= `pseudo.hi`, every other
candidate p <= `pseudo.lo`) give pseudo-positive and pseudo-negative pairs, mostly from
countries absent from training; the reranker continues training on them mixed with replayed
training pairs (models/reranker_pl/), re-scores train and test (scores_<split>_pl), and stage
2, the rule and the outputs are rebuilt under the tag "_pl".

Round 3, raw-text round (`--stage raw`, reranker (c), see README.md): a second reranker that reads
each record as shipped, "{business_name} | {business_address}" (`rerank.text: raw`: original
case, accents, punctuation, scripts, "@", ".com", brackets; the normalized view drops them).
It trains on `raw.train_s1` role-1 S1 (all of stage 1's sample), re-scores exactly the pairs
the main reranker scored, and stage 2 reads it alone (tag "_raw") or next to other rerankers (the variants
of run_pipeline.RAW_VARIANTS). A pilot during development (same 6,000 steps, same validation
pairs): log loss 0.0980 raw vs 0.1121 normalized. Blocking and the candidate set are unchanged.
`write_handoff` and `train_two_stage` build the ensemble member (d) of the final variant.

Steps (python -m src.run_pipeline --stage rerank --step <step> [--split train|test|all]):
  data    build and cache the training pairs (artifacts/rerank/train_pairs.parquet)
  time    200 training steps, projected full-run time
  train   full fine-tune (1 epoch) -> models/reranker/
  score   score the selected pairs of --split -> artifacts/rerank/scores_<split>.parquet
  pseudo  the pseudo-label fine-tune of round 2 -> models/reranker_pl/
"""

import math
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import torch

from .decide import groups
from .io import as_arrow, s1_roles
from .normalize.normalize import load_normalized
from .train import preds_dir
from .utils import device, log

COLUMNS = ["key", "country", "name_ascii", "address_norm"]


def rr_dir(cfg: dict):
    return cfg["paths"]["artifacts_dir"] / "rerank"


def model_dir(cfg: dict, tag: str = ""):
    return cfg["paths"]["models_dir"] / f"reranker{tag}"


def record_texts(df: pd.DataFrame) -> pa.Array:
    """'{name_ascii} | {address_norm}' for every record."""
    return as_arrow(df["name_ascii"] + " | " + df["address_norm"])


def split_texts(cfg: dict, split: str, side: str) -> pa.Array:
    """The reranker text of every record of one side ("s1" or "targets") of a split, in the
    row order pairs refer to: '{name_ascii} | {address_norm}' by default; with
    `rerank.text: raw` the record as shipped, '{business_name} | {business_address}',
    from the source files (targets = S2 rows, then S3 rows, as in the normalized tables)."""
    if cfg["rerank"].get("text", "norm") == "raw":
        from .io import load_source
        df = pd.concat([load_source(cfg, split, k, columns=["business_name", "business_address"])
                        for k in ((1,) if side == "s1" else (2, 3))], ignore_index=True)
        return as_arrow(df["business_name"] + " | " + df["business_address"])
    return record_texts(load_normalized(cfg, split, side, columns=COLUMNS))


def stage1_scores(cfg: dict, split: str) -> pd.DataFrame:
    """(s1, t, p[, label, emb_cos]) stage-1 probabilities of every scored pair of a split,
    sorted by (s1, t)."""
    if split == "train":
        from .stack import load_train_scored
        df = load_train_scored(cfg)
    else:
        from .train import score_test
        df = score_test(cfg)
    keep = [c for c in ("s1", "t", "p", "label", "emb_cos") if c in df]
    return df[keep]


def select_pairs(df: pd.DataFrame, r: dict) -> np.ndarray:
    """Rows to re-score (rows sorted by s1): stage-1 p >= score_low, each S1's score_top best
    by p, and its score_top_emb best by embedding cosine (when available)."""
    s1, p = df["s1"].to_numpy(), df["p"].to_numpy()
    starts, gid = groups(s1)

    def rank(v):
        order = np.lexsort((-v, gid))
        out = np.empty(len(v), np.int32)
        out[order] = np.arange(len(v)) - starts[gid[order]]
        return out

    keep = (p >= r["score_low"]) | (rank(p) < r.get("score_top", 1))
    if r.get("score_top_emb", 0) and "emb_cos" in df:
        keep |= rank(np.nan_to_num(df["emb_cos"].to_numpy(), nan=-1.0)) < r["score_top_emb"]
    return np.flatnonzero(keep)


def training_pairs(cfg: dict, tag: str = "") -> pd.DataFrame:
    """(s1, t, label[, split]) training pairs from a seeded sample of role-1 S1 records (cached;
    train_pairs<tag>.parquet, so the raw-text round's larger sample has its own file).

    With `rerank.pseudo_mix` (set by the raw round): pseudo-labeled TEST pairs (pseudo_pairs
    of a finished run's decisions) are mixed in, so one training sees the training countries
    and the unseen ones. The `split` column says which tables a row's s1/t index ("train" or
    "test"); rows without it are train rows.
    """
    path = rr_dir(cfg) / f"train_pairs{tag}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    out = _train_split_pairs(cfg)
    mix = cfg["rerank"].get("pseudo_mix")
    if mix:
        pl = pseudo_pairs(cfg, mix, mix.get("source_tag", ""))[["s1", "t", "label"]]
        out = pd.concat([out.assign(split="train"), pl.assign(split="test")], ignore_index=True)
        out = out.sample(frac=1.0, random_state=cfg["seed"]).reset_index(drop=True)
        log(f"reranker training pairs: {len(out):,} = {int((out['split'] == 'train').sum()):,} train + "
            f"{len(pl):,} pseudo-labeled test pairs ({int(pl['label'].sum()):,} positive)")
    rr_dir(cfg).mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return out


def _train_split_pairs(cfg: dict) -> pd.DataFrame:
    """(s1, t, label) of the seeded role-1 sample: every true pair plus sampled negatives."""
    r = cfg["rerank"]
    rng = np.random.default_rng(cfg["seed"])
    s1 = load_normalized(cfg, "train", "s1", columns=["key"])
    roles = s1_roles(cfg, s1["key"].to_numpy())
    keep = np.zeros(len(roles), bool)
    pool = np.flatnonzero(roles == 1)
    keep[rng.choice(pool, size=min(len(pool), r["train_s1"]), replace=False)] = True
    if cfg.get("prune", {}).get("enabled"):
        # hardness by the pruner's probability (out-of-sample for role 1), so the reranker
        # trains on the GPU while stage 1 is computed on the CPU
        from .io import truth_pairs
        from .prune import pruned_dir
        frames = []
        for part in sorted(pruned_dir(cfg, "train").glob("part-*.parquet")):
            c = pd.read_parquet(part, columns=["s1", "t", "prune_p"])
            frames.append(c[keep[c["s1"].to_numpy()]])
        df = pd.concat(frames, ignore_index=True).rename(columns={"prune_p": "p"})
        n_t = len(load_normalized(cfg, "train", "targets", columns=["key"]))
        tg_keys = load_normalized(cfg, "train", "targets", columns=["key"])["key"].to_numpy()
        ts, tt = truth_pairs(cfg, s1["key"].to_numpy(), tg_keys)
        true_codes = np.sort(ts.astype(np.int64) * n_t + tt)
        codes = df["s1"].to_numpy().astype(np.int64) * n_t + df["t"].to_numpy()
        pos = np.minimum(np.searchsorted(true_codes, codes), len(true_codes) - 1)
        df["label"] = (true_codes[pos] == codes).astype(np.int8)
    else:
        df = pd.read_parquet(preds_dir(cfg) / "train_oof.parquet", columns=["s1", "t", "label", "p"])  # role 1
        df = df[keep[df["s1"].to_numpy()]]
    out = sample_negatives(df, r["neg_per_pos"], rng)
    out = out.sample(frac=1.0, random_state=cfg["seed"]).reset_index(drop=True)
    log(f"reranker training pairs: {len(out):,} ({int(out['label'].sum()):,} positive) from "
        f"{out['s1'].nunique():,} role-1 S1")
    return out


def sample_negatives(df: pd.DataFrame, neg_per_pos: float, rng) -> pd.DataFrame:
    """Every positive of each S1 plus ~neg_per_pos negatives per positive (at least
    neg_per_pos for an S1 without positives): 75% the hardest by p, 25% at random."""
    df = df.assign(rand=rng.random(len(df)).astype(np.float32))
    pos = df[df["label"] == 1]
    neg = df[df["label"] == 0]
    n_pos = pos.groupby("s1").size()
    quota = (np.maximum(n_pos.reindex(neg["s1"].unique(), fill_value=0), 1) * neg_per_pos)
    n_hard = np.ceil(quota * 0.75).astype(int)
    neg = neg.sort_values(["s1", "p"], ascending=[True, False])
    neg["hard_rank"] = neg.groupby("s1").cumcount()
    hard = neg[neg["hard_rank"].to_numpy() < n_hard.reindex(neg["s1"]).to_numpy()]
    rest = neg[neg["hard_rank"].to_numpy() >= n_hard.reindex(neg["s1"]).to_numpy()]
    rest = rest.sort_values(["s1", "rand"])
    rest_rank = rest.groupby("s1").cumcount().to_numpy()
    n_rand = (quota - n_hard).reindex(rest["s1"]).to_numpy()
    rand = rest[rest_rank < n_rand]
    return pd.concat([pos, hard, rand])[["s1", "t", "label"]]


def pair_texts(cfg: dict, split: str, pairs: pd.DataFrame) -> tuple[list[str], list[str]]:
    """(S1 text, target text) of each pair of a split; with a `split` column (mixed
    training pairs of round 3) each row's texts come from its own split's tables, in row order."""
    if "split" in pairs and pairs["split"].nunique() > 1:
        a, b = np.empty(len(pairs), object), np.empty(len(pairs), object)
        for sp in pd.unique(pairs["split"]):
            rows = np.flatnonzero(pairs["split"].to_numpy() == sp)
            a[rows], b[rows] = pair_texts(cfg, sp, pairs.iloc[rows])
        return a.tolist(), b.tolist()
    split = pairs["split"].iat[0] if "split" in pairs and len(pairs) else split
    return (split_texts(cfg, split, "s1").take(pa.array(pairs["s1"].to_numpy())).to_pylist(),
            split_texts(cfg, split, "targets").take(pa.array(pairs["t"].to_numpy())).to_pylist())


def make_dataset(cfg: dict, a: list[str], b: list[str], labels: np.ndarray, tok):
    """Tokenized training dataset; the two records are swapped for a seeded half of the rows."""
    from datasets import Dataset

    swap = np.random.default_rng(cfg["seed"]).random(len(a)) < 0.5
    first = [y if w else x for x, y, w in zip(a, b, swap)]
    second = [x if w else y for x, y, w in zip(a, b, swap)]
    ds = Dataset.from_dict({"a": first, "b": second, "label": labels.astype(np.float32)})
    max_len = cfg["rerank"]["max_length"]

    def enc(batch):
        x = tok(batch["a"], batch["b"], truncation=True, max_length=max_len, return_token_type_ids=False)
        x["labels"] = [[float(y)] for y in batch["label"]]
        return x

    return ds.map(enc, batched=True, batch_size=10_000, remove_columns=["a", "b", "label"],
                  num_proc=min(8, max(1, (cfg.get("threads") or 2) // 4)))


def finetune(cfg: dict, max_steps: int | None = None, init=None, ds=None, out_dir=None,
             lr: float | None = None, tag: str = "") -> dict:
    """BCE on one logit, 1 epoch, warmup 0.05, weight decay 0.01, bf16 on GPU, SDPA attention,
    length-grouped batches (less padding), seed 42. Default: reranker (a) on the
    role-1 training pairs; `init`/`ds`/`out_dir`/`lr` serve the pseudo-label round, `tag` the
    raw-text round (models/reranker<tag>/, train_pairs<tag>.parquet)."""
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                              Trainer, TrainingArguments)

    r = cfg["rerank"]
    out_dir = out_dir or model_dir(cfg, tag)
    if max_steps is None and (out_dir / "config.json").exists():
        log(f"reranker already trained in {out_dir}; skipped")
        return {"saved_to": str(out_dir)}
    src = str(init or r["model"])
    tok = AutoTokenizer.from_pretrained(src)
    if ds is None:
        pairs = training_pairs(cfg, tag)
        frac = float(r.get("train_frac", 1.0))
        if frac < 1.0:  # time fallback (run_v4.sh raw): a seeded fraction of the train rows, every pseudo row
            is_train = pairs["split"].to_numpy() == "train" if "split" in pairs else np.ones(len(pairs), bool)
            drop = is_train & (np.random.default_rng(cfg["seed"] + 3).random(len(pairs)) >= frac)
            pairs = pairs[~drop].reset_index(drop=True)
            log(f"reranker: train_frac {frac} -> {len(pairs):,} training pairs")
        a, b = pair_texts(cfg, "train", pairs)
        ds = make_dataset(cfg, a, b, pairs["label"].to_numpy(), tok)
        del a, b
    cuda = torch.cuda.is_available()
    kw = {"attn_implementation": "sdpa"} if cuda else {}
    if init is None:  # a fresh one-logit head on the pretrained encoder
        kw["ignore_mismatched_sizes"] = True
    model = AutoModelForSequenceClassification.from_pretrained(
        src, num_labels=1, problem_type="multi_label_classification", **kw)
    args = TrainingArguments(
        output_dir=str(out_dir), num_train_epochs=1, per_device_train_batch_size=r["batch_size"],
        learning_rate=lr or r["lr"], warmup_ratio=0.05, weight_decay=0.01, bf16=cuda, seed=cfg["seed"],
        save_strategy="no", logging_steps=100, report_to="none", max_steps=max_steps or -1,
        group_by_length=True, dataloader_num_workers=4 if cuda else 0,
        optim="adamw_torch_fused" if cuda else "adamw_torch", tf32=cuda or None)
    trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=DataCollatorWithPadding(tok))
    t0 = time.perf_counter()
    trainer.train()
    seconds = time.perf_counter() - t0
    steps_full = math.ceil(len(ds) / r["batch_size"])
    info = {"pairs": len(ds), "steps_run": max_steps or steps_full, "steps_full": steps_full,
            "seconds": round(seconds, 1),
            "projected_full_minutes": round(seconds / (max_steps or steps_full) * steps_full / 60, 1),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if cuda else 0}
    if max_steps is None:
        tmp = out_dir.with_name(out_dir.name + "_partial")
        model.save_pretrained(str(tmp))
        tok.save_pretrained(str(tmp))
        if out_dir.exists():
            import shutil
            shutil.rmtree(out_dir)
        tmp.replace(out_dir)
        info["saved_to"] = str(out_dir)
    log(f"reranker fine-tune: {info}")
    return info


class ScoreBatches(torch.utils.data.Dataset):
    """Tokenized scoring batches: batch i = pairs order[i*bs:(i+1)*bs]. Built in DataLoader
    worker processes, so tokenizing the next batches overlaps the GPU's work on this
    one; in the main process, tokenizing held scoring at ~1,500 pairs/s on an L4. The
    batches, and so the scores, are exactly those of the one-process loop."""

    def __init__(self, a: pa.Array, b: pa.Array, order: np.ndarray, bs: int, tok, max_len: int):
        self.a, self.b, self.order, self.bs, self.tok, self.max_len = a, b, order, bs, tok, max_len

    def __len__(self) -> int:
        return (len(self.order) + self.bs - 1) // self.bs

    def __getitem__(self, i: int):
        idx = self.order[i * self.bs:(i + 1) * self.bs]
        x = self.tok(self.a.take(pa.array(idx)).to_pylist(), self.b.take(pa.array(idx)).to_pylist(),
                     truncation=True, max_length=self.max_len, padding=True, return_token_type_ids=False,
                     return_tensors="pt")  # ModernBERT takes no token type ids
        return idx, dict(x)


def score_texts(cfg: dict, a: pa.Array, b: pa.Array, label: str = "", tag: str = "") -> np.ndarray:
    """Reranker P(match) of text pairs (a[i], b[i]), S1 record first; batched by text length
    to keep padding low, tokenized by `rerank.score_workers` worker processes on GPU."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    r = cfg["rerank"]
    dev = device()
    order = np.argsort(pc.utf8_length(a).to_numpy() + pc.utf8_length(b).to_numpy(), kind="stable")
    tok = AutoTokenizer.from_pretrained(str(model_dir(cfg, tag)))
    kw = {"attn_implementation": "sdpa", "torch_dtype": torch.bfloat16} if dev == "cuda" else {}
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir(cfg, tag)), **kw).to(dev).eval()
    out = np.empty(len(a), np.float32)
    bs, t0 = r["score_batch"], time.perf_counter()
    workers = int(r.get("score_workers", 6)) if dev == "cuda" else 0
    loader = torch.utils.data.DataLoader(ScoreBatches(a, b, order, bs, tok, r["max_length"]), batch_size=None,
                                         num_workers=workers, pin_memory=dev == "cuda",
                                         prefetch_factor=4 if workers else None)
    done = 0
    with torch.inference_mode():
        for i, (idx, x) in enumerate(loader):
            idx = np.asarray(idx)
            x = {k: v.to(dev, non_blocking=True) for k, v in x.items()}
            out[idx] = torch.sigmoid(model(**x).logits.float()[:, 0]).cpu().numpy()
            done += len(idx)
            if i % 2000 == 1999:
                log(f"reranker {label}: {done:,}/{len(a):,} pairs ({done / (time.perf_counter() - t0):.0f}/s)")
    del model
    if dev == "cuda":
        torch.cuda.empty_cache()
    return out


def selected_pairs(cfg: dict, split: str) -> pd.DataFrame:
    """(s1, t) of the pairs the reranker scores on a split (see score_split): every S1's
    selection by select_pairs; on train only the S1 stage 2 needs (plus their competitors)."""
    from .stack import train_s1_mask

    r = cfg["rerank"]
    df = stage1_scores(cfg, split)
    rows = select_pairs(df, r)  # per-S1 selection over every S1 of the split
    if split == "train":  # stage 2's training sample and validation (plus per-country samples if asked)
        s1 = load_normalized(cfg, "train", "s1", columns=["key", "country"])
        roles = s1_roles(cfg, s1["key"].to_numpy())
        need = (roles == 2) | train_s1_mask(cfg, roles)
        if r.get("score_loco"):
            country = s1["country"].to_numpy(dtype=object)
            for c in pd.unique(country):
                need |= train_s1_mask(cfg, roles, country, c)
        s_sel, t_sel = df["s1"].to_numpy()[rows], df["t"].to_numpy()[rows]
        keep = need[s_sel]
        n_own = int(keep.sum())
        if r.get("complete_competition"):
            contested = np.zeros(int(df["t"].max()) + 1, bool)
            contested[t_sel[keep]] = True
            keep |= contested[t_sel]
        rows = rows[keep]
        log(f"reranker [train]: {n_own:,} pairs of the needed S1 + {len(rows) - n_own:,} competitor pairs")
    return df.iloc[rows][["s1", "t"]].reset_index(drop=True)


def write_handoff(cfg: dict) -> None:
    """Write the raw round's pair lists (handoff_pairs_<split>.parquet) plus its training
    pairs, so member (d) is trained and scored on exactly the same pairs, on this or a second
    GPU machine (`--stage raw --step handoff`; README.md)."""
    training_pairs(cfg, "_raw")
    for split in ("train", "test"):
        path = rr_dir(cfg) / f"handoff_pairs_{split}.parquet"
        if path.exists():
            log(f"{path.name} exists; kept")
            continue
        pairs = selected_pairs(cfg, split)
        pairs.to_parquet(path, index=False, compression="zstd")
        log(f"wrote {path.name}: {len(pairs):,} pairs of {pairs['s1'].nunique():,} S1")


def train_two_stage(cfg: dict, tag: str = "_raw2") -> None:
    """Member (d), the second raw reranker (`--stage raw --step train2`): the
    two-stage recipe instead of the mixed one, for a diverse ensemble member. Stage A: one
    epoch on the train-split rows of the raw round's training pairs (`raw.second.train_frac`
    of them); stage B: continue on the pseudo-labeled rows plus replayed train rows at a low
    learning rate (as the pseudo-label round). `raw.second` sets seed, batch and rates."""
    from transformers import AutoTokenizer

    sec = cfg["raw"]["second"]
    c = dict(cfg)
    c["seed"] = int(sec.get("seed", cfg["seed"]))
    c["rerank"] = {**cfg["rerank"], "batch_size": sec.get("batch_size", cfg["rerank"]["batch_size"]),
                   "lr": sec.get("lr", cfg["rerank"]["lr"])}
    out_dir = model_dir(cfg, tag)
    if (out_dir / "config.json").exists():
        log(f"second reranker already trained in {out_dir}; skipped")
        return
    pairs = training_pairs(cfg, "_raw")
    if "split" not in pairs:
        raise SystemExit("train2 needs the raw round's mixed training pairs (raw.pseudo.enabled)")
    rng = np.random.default_rng(c["seed"])
    tr = pairs[pairs["split"] == "train"]
    tr = tr[rng.random(len(tr)) < float(sec.get("train_frac", 1.0))].reset_index(drop=True)
    pl = pairs[pairs["split"] == "test"].reset_index(drop=True)
    tok = AutoTokenizer.from_pretrained(str(cfg["rerank"]["model"]))
    stage_a = out_dir.with_name(out_dir.name + "_stageA")
    if not (stage_a / "config.json").exists():
        a, b = pair_texts(c, "train", tr)
        log(f"second reranker stage A: {len(tr):,} train pairs, seed {c['seed']}, batch {c['rerank']['batch_size']}")
        finetune(c, ds=make_dataset(c, a, b, tr["label"].to_numpy(), tok), out_dir=stage_a)
        del a, b
    replay = tr.sample(n=min(len(tr), int(len(pl) * float(sec.get("replay", 0.3)))), random_state=c["seed"])
    mixed = pd.concat([pl, replay], ignore_index=True).sample(frac=1.0, random_state=c["seed"]).reset_index(drop=True)
    a, b = pair_texts(c, "train", mixed)
    log(f"second reranker stage B: {len(pl):,} pseudo pairs + {len(replay):,} replayed train pairs, lr {sec.get('lr_pseudo')}")
    finetune(c, init=stage_a, ds=make_dataset(c, a, b, mixed["label"].to_numpy(), tok), out_dir=out_dir,
             lr=float(sec.get("lr_pseudo", cfg["pseudo"]["lr"])))


def score_split(cfg: dict, split: str, tag: str = "") -> None:
    """Score the selected pairs of a split with the fine-tuned reranker (S1 record first);
    cached as parquet (skipped when the cache matches the model and the selection).

    Train: the pairs of stage 2's training sample and of validation, plus (with
    `rerank.complete_competition`) every selected pair of any other train S1 that competes
    for one of their targets. Stage 2's strongest feature compares a pair's reranker score
    with the best score another S1 has for the same target; on test every S1 is scored, so
    without these competitor pairs the train/validation feature saw only ~1/3 of the
    competition. Pairs already scored by the same model are reused, not re-scored.

    Handoff: when artifacts/rerank/handoff_pairs_<split>.parquet (columns s1, t) exists,
    exactly those pairs are scored, so a second GPU machine without the stage-1 artifacts can
    score member (d) (the file is the s1/t columns of the main run's scores_<split>).
    """
    import json

    from .stack import train_s1_mask

    r = cfg["rerank"]
    path = rr_dir(cfg) / f"scores_{split}{tag}.parquet"
    model_mtime = (model_dir(cfg, tag) / "config.json").stat().st_mtime_ns
    ident = {"model_mtime": model_mtime,
             "selection": {k: r.get(k) for k in ("score_low", "score_top", "score_top_emb")},
             "stack_train_s1": cfg["stack"]["train_s1"]}
    if split == "train" and r.get("complete_competition"):
        ident["complete_competition"] = True  # train only: the test stamp stays as it was
    if r.get("text", "norm") != "norm":
        ident["text"] = r["text"]  # only when set, so the main run's stamps stay valid
    if r.get("tta_band"):
        ident["tta_band"] = list(r["tta_band"])
    stamp = path.with_suffix(".json")
    if path.exists() and stamp.exists() and json.loads(stamp.read_text()) == ident:
        log(f"reranker scores [{split}{tag}] up to date; skipped")
        return
    handoff = rr_dir(cfg) / f"handoff_pairs_{split}.parquet"
    if handoff.exists():
        pairs = pd.read_parquet(handoff, columns=["s1", "t"])
        pairs = pairs.iloc[np.lexsort((pairs["t"].to_numpy(), pairs["s1"].to_numpy()))].reset_index(drop=True)
        log(f"reranker [{split}{tag}]: {len(pairs):,} pairs from {handoff.name}")
    else:
        pairs = selected_pairs(cfg, split)
    n_t = int(pairs["t"].max()) + 1 if len(pairs) else 1
    codes = pairs["s1"].to_numpy().astype(np.int64) * n_t + pairs["t"].to_numpy()
    out = np.full(len(pairs), np.nan, np.float32)
    old_stamp = json.loads(stamp.read_text()) if stamp.exists() else {}
    if path.exists() and old_stamp.get("model_mtime") == model_mtime:  # reuse this model's scores
        old = pd.read_parquet(path)
        old_codes = old["s1"].to_numpy().astype(np.int64) * n_t + old["t"].to_numpy()
        ok = old["t"].to_numpy() < n_t
        order = np.argsort(old_codes[ok])
        oc, op = old_codes[ok][order], old["rr_p"].to_numpy()[ok][order]
        pos = np.minimum(np.searchsorted(oc, codes), max(len(oc) - 1, 0))
        hit = (oc[pos] == codes) if len(oc) else np.zeros(len(codes), bool)
        out[hit] = op[pos[hit]]
        log(f"reranker [{split}{tag}]: {int(hit.sum()):,} of {len(pairs):,} pairs reused from the cache")
        del old
    todo = np.flatnonzero(np.isnan(out))
    t0 = time.perf_counter()
    if len(todo):
        ta = split_texts(cfg, split, "s1").take(pa.array(pairs["s1"].to_numpy()[todo]))
        tb = split_texts(cfg, split, "targets").take(pa.array(pairs["t"].to_numpy()[todo]))
        out[todo] = score_texts(cfg, ta, tb, label=split, tag=tag)
        band = r.get("tta_band")
        if band:  # uncertain pairs also scored with the records swapped (the model trained
            # on both orders); the two probabilities are averaged
            pos = np.flatnonzero((out[todo] > band[0]) & (out[todo] < band[1]))
            if len(pos):
                idx = todo[pos]
                swapped = score_texts(cfg, tb.take(pa.array(pos)), ta.take(pa.array(pos)), label=f"{split} swapped", tag=tag)
                out[idx] = 0.5 * (out[idx] + swapped)
                log(f"reranker [{split}{tag}]: {len(pos):,} uncertain pairs ({len(pos) / len(todo):.1%}) "
                    f"averaged over both record orders")
        del ta, tb
    rr_dir(cfg).mkdir(parents=True, exist_ok=True)
    pairs.assign(rr_p=out).to_parquet(path.with_name(path.name + ".partial"), index=False)
    path.with_name(path.name + ".partial").replace(path)
    stamp.write_text(json.dumps(ident))
    log(f"reranker [{split}{tag}]: {len(pairs):,} pairs of {pairs['s1'].nunique():,} S1 ready "
        f"({len(todo):,} scored now in {time.perf_counter() - t0:.0f}s)")


# ---- optional pseudo-label round (pseudo.enabled / --step pseudo) ---------------------------

def pseudo_pairs(cfg: dict, pc_: dict | None = None, source_tag: str = "") -> pd.DataFrame:
    """(s1, t, label, country) pseudo-labeled test pairs from a finished run's decisions.

    An S1 qualifies when its decision is unambiguous under stage 2: it emits at least one
    pair, every emitted pair has p >= pseudo.hi and every other candidate p <= pseudo.lo;
    or it emits nothing and its best candidate has p <= pseudo.lo_single (a confident
    singleton, all pairs negative). Countries absent from training get up to
    pseudo.unseen_s1 such S1 each, the others pseudo.seen_s1.

    `pc_` (default cfg["pseudo"]) holds those settings; `source_tag` names the run whose
    decisions are used (preds/test<source_tag>.parquet and its thresholds): "" = the main
    run (round 2), "_pl" = round 2's own decisions (round 3: a second, larger self-training
    round from the better model).
    """
    import json

    from .decide import emit_mask, thresholds_path

    pc_ = pc_ or cfg["pseudo"]
    df = pd.read_parquet(preds_dir(cfg) / f"test{source_tag}.parquet", columns=["s1", "t", "p"])
    df = df.iloc[np.lexsort((df["t"].to_numpy(), df["s1"].to_numpy()))].reset_index(drop=True)
    source = {**cfg, "stack": {**cfg["stack"], "rerank_tag": source_tag}}
    th = json.loads(thresholds_path(source).read_text())["thresholds"]
    emit = emit_mask(df, th)
    s1, p = df["s1"].to_numpy(), df["p"].to_numpy()
    starts, gid = groups(s1)
    n_emit = np.add.reduceat(emit.astype(np.int32), starts)
    min_emit = np.minimum.reduceat(np.where(emit, p, 1.0), starts)
    max_rest = np.maximum.reduceat(np.where(emit, 0.0, p), starts)
    best = np.maximum.reduceat(p, starts)
    clean = ((n_emit > 0) & (min_emit >= pc_["hi"]) & (max_rest <= pc_["lo"])) | \
            ((n_emit == 0) & (best <= pc_["lo_single"]))
    test_c = load_normalized(cfg, "test", "s1", columns=["country"])["country"].to_numpy(dtype=object)
    train_countries = set(pd.unique(load_normalized(cfg, "train", "s1", columns=["country"])["country"]))
    g_s1 = s1[starts]
    rng = np.random.default_rng(cfg["seed"] + 7)
    chosen = np.zeros(len(starts), bool)
    report = {}
    for c in pd.unique(test_c[g_s1]):
        cand = np.flatnonzero(clean & (test_c[g_s1] == c))
        cap = pc_["seen_s1"] if c in train_countries else pc_["unseen_s1"]
        pick = rng.choice(cand, size=min(cap, len(cand)), replace=False) if len(cand) else cand
        chosen[pick] = True
        report[str(c)] = {"clean_s1": int(len(cand)), "of_s1": int((test_c[g_s1] == c).sum()),
                          "used": int(len(pick)), "seen_in_train": c in train_countries}
    rows = chosen[gid]
    out = pd.DataFrame({"s1": s1[rows], "t": df["t"].to_numpy()[rows], "p": p[rows],
                        "label": emit[rows].astype(np.int8)})
    out = sample_negatives(out, pc_["neg_per_pos"], rng)
    out["country"] = test_c[out["s1"].to_numpy()]
    log(f"pseudo-label pairs{source_tag}: {len(out):,} ({int(out['label'].sum()):,} positive); per country: {report}")
    rr_dir(cfg).mkdir(parents=True, exist_ok=True)
    (rr_dir(cfg) / f"pseudo_report{cfg['stack'].get('rerank_tag', '')}.json").write_text(json.dumps(report, indent=1))
    return out


def pseudo_finetune(cfg: dict) -> None:
    """Continue training the reranker on pseudo-labeled test pairs plus replayed training
    pairs (models/reranker_pl/)."""
    from transformers import AutoTokenizer

    pc_ = cfg["pseudo"]
    out_dir = model_dir(cfg, "_pl")
    if (out_dir / "config.json").exists():
        log(f"pseudo-label reranker already trained in {out_dir}; skipped")
        return
    tok = AutoTokenizer.from_pretrained(str(model_dir(cfg)))
    pl = pseudo_pairs(cfg)
    a, b = pair_texts(cfg, "test", pl)
    replay = training_pairs(cfg).sample(n=min(len(training_pairs(cfg)), int(len(pl) * pc_["replay"])),
                                        random_state=cfg["seed"])
    ra, rb = pair_texts(cfg, "train", replay)
    labels = np.concatenate([pl["label"].to_numpy(), replay["label"].to_numpy()])
    order = np.random.default_rng(cfg["seed"]).permutation(len(labels))
    a, b = np.array(a + ra, dtype=object)[order].tolist(), np.array(b + rb, dtype=object)[order].tolist()
    ds = make_dataset(cfg, a, b, labels[order], tok)
    finetune(cfg, init=model_dir(cfg), ds=ds, out_dir=out_dir, lr=pc_["lr"])


def run(cfg: dict, step: str, split: str) -> None:
    """Dispatch one reranker step."""
    tag = "_pl" if step in ("pseudo", "score_pl") else ""
    if step == "data":
        training_pairs(cfg)
    elif step == "time":
        finetune(cfg, max_steps=200)
    elif step == "train":
        finetune(cfg)
    elif step == "pseudo":
        pseudo_finetune(cfg)
    elif step in ("score", "score_pl"):
        for sp in (("train", "test") if split == "all" else (split,)):
            score_split(cfg, sp, tag)
    else:
        raise SystemExit(f"unknown rerank step '{step}'")
