"""Model 2: Snowflake/snowflake-arctic-embed-m-v2.0 as blocker B4 (bi-encoder).

Text per record: "query: {name_ascii} | {address_norm}" on both sides (the model card's
query prompt; this is symmetric record-to-record matching). Embeddings are truncated to
256 dims (Matryoshka), L2-normalized and stored as float16 memory-mapped .npy files.

Loading (verified): the model's custom GTE code needs transformers 4.x, and its config
turns on xformers attention and unpadding; both are switched off here so PyTorch SDPA
is used (xformers is not an approved library).

Search: exact inner product on the GPU per country partition: targets sit on the GPU in
fp16, queries go in chunks, and a running top-k is kept. An HNSW index over ~10M
256-d float vectors would need ~13 GB of RAM, more than this machine has free.

Steps (python -m src.run_pipeline --stage embed --step <step>):
  baseline  recall@20 of the un-tuned model on 20k validation S1 vs all train targets
  time      200 training steps, then the projected full-run time
  train     full fine-tune (1 epoch), saved to models/embedder_ft/
  eval      recall@20 of the fine-tuned model on the same 20k validation S1
  index     encode every S1 and target of --split, top-K per S1 -> B4 candidates
"""

import json
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import torch

from .blocking import candidates_dir, coverage, query_rows
from .io import as_arrow, s1_roles, truth_pairs
from .normalize.normalize import load_normalized
from .utils import device, log

PROMPT = "query: "
LOAD_KWARGS = {"use_memory_efficient_attention": False, "unpad_inputs": False}
COLUMNS = ["key", "country", "name_ascii", "address_norm"]


def emb_dir(cfg: dict):
    return cfg["paths"]["artifacts_dir"] / "embeddings"


def ft_dir(cfg: dict):
    return cfg["paths"]["models_dir"] / "embedder_ft"


def load_embedder(cfg: dict, path=None, bf16: bool = True):
    """SentenceTransformer on the GPU with SDPA attention and the configured max length."""
    from sentence_transformers import SentenceTransformer

    dev = device()
    model = SentenceTransformer(str(path or cfg["embed"]["model"]), trust_remote_code=True,
                                device=dev, config_kwargs=LOAD_KWARGS)
    model.max_seq_length = cfg["embed"]["max_seq_length"]
    return model.to(torch.bfloat16) if bf16 and dev == "cuda" else model


def record_texts(df: pd.DataFrame) -> pa.Array:
    """'query: {name_ascii} | {address_norm}' for every record."""
    return as_arrow(PROMPT + df["name_ascii"] + " | " + df["address_norm"])


def encode_array(model, texts: list[str], dim: int, batch_size: int) -> np.ndarray:
    """In-memory encoding (small universes): L2-normalized float16 [n, dim]."""
    with torch.inference_mode():
        e = model.encode(texts, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=False)
    return torch.nn.functional.normalize(e[:, :dim].float(), dim=1).half().cpu().numpy()


def embed_universe(cfg: dict, s1: pd.DataFrame, tg: pd.DataFrame, queries: np.ndarray):
    """B4 results and embedding-feature arrays for an in-memory universe (stress set)."""
    e = cfg["embed"]
    tag = cfg["blocking"]["b4"]
    model = load_embedder(cfg, ft_dir(cfg) if tag == "ft" else None)
    s1_emb = encode_array(model, record_texts(s1).to_pylist(), e["dim"], e["encode_batch"])
    t_emb = encode_array(model, record_texts(tg).to_pylist(), e["dim"], e["encode_batch"])
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    s1_c, t_c = s1["country"].to_numpy(dtype=object), tg["country"].to_numpy(dtype=object)
    b4 = search(s1_emb, t_emb, queries, s1_c, t_c, cfg["blocking"]["k_max"])
    rev = reverse_search(s1_emb, t_emb, s1_c, t_c)
    emb = {"s1": s1_emb, "t": t_emb, "rev_best": rev["best"], "rev_second": rev["second"],
           "rev_best_s1": rev["best_s1"]}
    return b4, emb


def encode(model, texts: pa.Array, path, dim: int, batch_size: int, chunk: int = 200_000) -> np.ndarray:
    """Encode texts to an L2-normalized float16 [n, dim] .npy (cached; written atomically)."""
    if path.exists():
        return np.load(path, mmap_mode="r")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".partial.npy")
    out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(len(texts), dim))
    t0 = time.perf_counter()
    for a in range(0, len(texts), chunk):
        batch = texts.slice(a, chunk).to_pylist()
        with torch.inference_mode():
            e = model.encode(batch, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=False)
        e = torch.nn.functional.normalize(e[:, :dim].float(), dim=1)
        out[a:a + len(batch)] = e.half().cpu().numpy()
        if (a // chunk) % 10 == 9:
            done = a + len(batch)
            log(f"encoded {done:,}/{len(texts):,} ({done / (time.perf_counter() - t0):.0f}/s)")
    out.flush()
    del out
    tmp.replace(path)
    return np.load(path, mmap_mode="r")


def gpu_topk(q: np.ndarray, t: np.ndarray, k: int, q_chunk: int = 1024,
             t_chunk: int = 1_000_000) -> tuple[np.ndarray, np.ndarray]:
    """Exact top-k inner products of every query row against all target rows, on the GPU
    (fp16), or on the CPU in fp32 when no GPU is visible (smoke tests)."""
    dev = device()
    dtype = torch.float16 if dev == "cuda" else torch.float32
    T = torch.from_numpy(np.ascontiguousarray(t)).to(dev, dtype)
    k = min(k, len(t))
    idx = np.empty((len(q), k), dtype=np.int32)
    val = np.empty((len(q), k), dtype=np.float32)
    with torch.inference_mode():
        for a in range(0, len(q), q_chunk):
            Q = torch.from_numpy(np.ascontiguousarray(q[a:a + q_chunk])).to(dev, dtype)
            best_s = best_i = None
            for b in range(0, len(T), t_chunk):
                s, i = torch.topk(Q @ T[b:b + t_chunk].T, min(k, len(T) - b), dim=1)
                i = i + b
                if best_s is None:
                    best_s, best_i = s, i
                else:
                    cs, ci = torch.cat([best_s, s], 1), torch.cat([best_i, i], 1)
                    best_s, j = torch.topk(cs, k, dim=1)
                    best_i = torch.gather(ci, 1, j)
            idx[a:a + len(Q)] = best_i.int().cpu().numpy()
            val[a:a + len(Q)] = best_s.float().cpu().numpy()
    del T
    if dev == "cuda":
        torch.cuda.empty_cache()
    return idx, val


def search(q_emb, t_emb, q_rows, s1_country, t_country, k) -> dict[str, np.ndarray]:
    """Top-k targets per query S1 row within its country: flat s1, t, score, rank arrays."""
    out = {"s1": [], "t": [], "score": [], "rank": []}
    for c in pd.unique(s1_country[q_rows]):
        qr = q_rows[s1_country[q_rows] == c]
        tr = np.flatnonzero(t_country == c)
        if len(tr) == 0:
            continue
        idx, val = gpu_topk(q_emb[qr], np.asarray(t_emb[tr]), k)
        out["s1"].append(np.repeat(qr, idx.shape[1]).astype(np.int32))
        out["t"].append(tr[idx.ravel()].astype(np.int32))
        out["score"].append(val.ravel())
        out["rank"].append(np.tile(np.arange(1, idx.shape[1] + 1, dtype=np.int16), len(qr)))
    return {key: np.concatenate(v) for key, v in out.items()}


def eval_queries(cfg: dict, s1: pd.DataFrame) -> np.ndarray:
    """The fixed, seeded sample of validation S1 rows used for recall@k."""
    roles = s1_roles(cfg, s1["key"].to_numpy())
    rng = np.random.default_rng(cfg["seed"])
    return np.sort(rng.choice(np.flatnonzero(roles == 2), cfg["embed"]["eval_queries"], replace=False))


def recall_eval(cfg: dict, tag: str, path=None) -> dict:
    """recall@k (pair and entity) on the fixed validation sample, overall and per country."""
    e = cfg["embed"]
    s1 = load_normalized(cfg, "train", "s1", columns=COLUMNS)
    tg = load_normalized(cfg, "train", "targets", columns=COLUMNS)
    model = load_embedder(cfg, path)
    q = eval_queries(cfg, s1)
    t_emb = encode(model, record_texts(tg), emb_dir(cfg) / f"train_targets_{tag}.npy", e["dim"], e["encode_batch"])
    q_emb_small = encode(model, record_texts(s1.iloc[q]), emb_dir(cfg) / f"eval_s1_{tag}.npy", e["dim"], e["encode_batch"])
    q_emb = np.zeros((len(s1), e["dim"]), dtype=np.float16)  # indexable by S1 row
    q_emb[q] = q_emb_small
    del model
    torch.cuda.empty_cache()
    r = search(q_emb, t_emb, q, s1["country"].to_numpy(dtype=object), tg["country"].to_numpy(dtype=object),
               e["eval_k"])
    ts, tt = truth_pairs(cfg, s1["key"].to_numpy(), tg["key"].to_numpy())
    country = s1["country"].to_numpy(dtype=object)
    result = {"tag": tag, "k": e["eval_k"], "queries": len(q)}
    for name, rows in [("overall", q)] + [(c, q[country[q] == c]) for c in pd.unique(country[q])]:
        cov = coverage(r["s1"], r["t"], ts, tt, rows, len(s1), len(tg))
        result[name] = {"pair_recall": cov["pair_recall"], "entity_recall": cov["entity_recall_ceiling"]}
    out = emb_dir(cfg) / f"recall_{tag}.json"
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    log(f"recall@{e['eval_k']} [{tag}]: " + json.dumps({k: v for k, v in result.items() if isinstance(v, dict)}))
    return result


def training_rows(cfg: dict):
    """(anchor, positive, negative) texts for the embedder's own S1 sample (role 3).

    One row per ground-truth pair; the negative is the S1 record's best-ranked candidate
    from the lexical blockers (B2w, then B1) that is not a true match. Role 3 is disjoint
    from stage 1's training S1 (role 1), validation and stage 2's sample, so every cosine
    those see is out-of-sample.
    """
    from datasets import Dataset

    from .blocking import lexical_negatives

    s1 = load_normalized(cfg, "train", "s1", columns=COLUMNS)
    tg = load_normalized(cfg, "train", "targets", columns=COLUMNS)
    roles = s1_roles(cfg, s1["key"].to_numpy())
    role = cfg["embed"].get("train_role", 3)
    ts, tt = truth_pairs(cfg, s1["key"].to_numpy(), tg["key"].to_numpy())
    keep = roles[ts] == role
    ps, pt = ts[keep], tt[keep]
    true_codes = np.sort(ps.astype(np.int64) * len(tg) + pt)
    c = lexical_negatives(cfg, np.flatnonzero(roles == role))
    codes = c["s1"].to_numpy().astype(np.int64) * len(tg) + c["t"].to_numpy()
    pos = np.minimum(np.searchsorted(true_codes, codes), max(len(true_codes) - 1, 0))
    c = c[true_codes[pos] != codes]
    r2 = np.where(c["b2w_rank"] > 0, c["b2w_rank"], 999).astype(np.int32) if "b2w_rank" in c else 999
    r1 = np.where(c["b1_rank"] > 0, c["b1_rank"], 999).astype(np.int32) if "b1_rank" in c else 999
    c = c.assign(order=r2 * 1000 + r1).sort_values(["s1", "order"]).drop_duplicates("s1")
    neg = c.set_index("s1")["t"]
    has_neg = np.isin(ps, neg.index.to_numpy())
    ps, pt = ps[has_neg], pt[has_neg]
    rng = np.random.default_rng(cfg["seed"])
    if len(ps) > cfg["embed"]["train_rows_max"]:
        uniq = np.unique(ps)
        n_keep = int(len(uniq) * cfg["embed"]["train_rows_max"] / len(ps))
        chosen = np.isin(ps, rng.choice(uniq, n_keep, replace=False))
        ps, pt = ps[chosen], pt[chosen]
    s_text, t_text = record_texts(s1), record_texts(tg)
    ds = Dataset.from_dict({"anchor": s_text.take(pa.array(ps)).to_pylist(),
                            "positive": t_text.take(pa.array(pt)).to_pylist(),
                            "negative": t_text.take(pa.array(neg.loc[ps].to_numpy())).to_pylist()})
    log(f"embedder training rows: {len(ds):,} from {len(np.unique(ps)):,} role-{role} S1")
    return ds.shuffle(seed=cfg["seed"])


def finetune(cfg: dict, max_steps: int | None = None) -> dict:
    """Fine-tune the bi-encoder: CachedMNRL, effective batch 256, NO_DUPLICATES, lr 2e-5, 1 epoch, bf16."""
    from sentence_transformers import (SentenceTransformerTrainer, SentenceTransformerTrainingArguments,
                                       losses)
    from sentence_transformers.training_args import BatchSamplers

    e = cfg["embed"]
    if max_steps is None and (ft_dir(cfg) / "config.json").exists():
        log(f"embedder already fine-tuned in {ft_dir(cfg)}; skipped")
        return {"saved_to": str(ft_dir(cfg))}
    ds = training_rows(cfg)
    model = load_embedder(cfg, bf16=False)  # fp32 weights, bf16 autocast in training
    loss = losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size=e["mini_batch_size"])
    cuda = torch.cuda.is_available()
    args = SentenceTransformerTrainingArguments(
        output_dir=str(ft_dir(cfg)), num_train_epochs=1, per_device_train_batch_size=e["batch_size"],
        learning_rate=e["lr"], warmup_ratio=0.05, bf16=cuda, seed=cfg["seed"],
        batch_sampler=BatchSamplers.NO_DUPLICATES, save_strategy="no", logging_steps=100,
        max_steps=max_steps or -1, report_to="none", dataloader_num_workers=2 if cuda else 0,
        optim="adamw_torch_fused" if cuda else "adamw_torch")
    trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=ds, loss=loss)
    t0 = time.perf_counter()
    trainer.train()
    seconds = time.perf_counter() - t0
    steps_full = len(ds) // e["batch_size"]
    info = {"rows": len(ds), "steps_run": max_steps or steps_full, "steps_full": steps_full,
            "seconds": round(seconds, 1),
            "projected_full_minutes": round(seconds / (max_steps or steps_full) * steps_full / 60, 1),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if cuda else 0}
    if max_steps is None:
        model.save_pretrained(str(ft_dir(cfg)))
        info["saved_to"] = str(ft_dir(cfg))
    log(f"embedder fine-tune: {info}")
    return info


def index(cfg: dict, split: str, path=None, tag: str = "ft") -> None:
    """Encode all records of a split, search top-K per S1 within countries, cache as B4 results
    (and each target's best S1 by reverse search). Every output is cached; a finished split
    is skipped."""
    e = cfg["embed"]
    s1 = load_normalized(cfg, split, "s1", columns=COLUMNS)
    _, q = query_rows(cfg, split, s1["key"].to_numpy())
    out = candidates_dir(cfg, split)
    b4_path = out / f"_b4_{tag}_q{len(q)}_k{cfg['blocking']['k_max']}.parquet"
    rev_path = emb_dir(cfg) / f"{split}_reverse_{tag}.npz"
    if b4_path.exists() and rev_path.exists():
        log(f"B4 [{split}]: already indexed; skipped")
        return
    tg = load_normalized(cfg, split, "targets", columns=COLUMNS)
    t_path, q_path = emb_dir(cfg) / f"{split}_targets_{tag}.npy", emb_dir(cfg) / f"{split}_s1_{tag}.npy"
    if not (t_path.exists() and q_path.exists()):
        model = load_embedder(cfg, path)
        t_emb = encode(model, record_texts(tg), t_path, e["dim"], e["encode_batch"])
        q_emb = encode(model, record_texts(s1), q_path, e["dim"], e["encode_batch"])
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    t_emb, q_emb = np.load(t_path, mmap_mode="r"), np.load(q_path, mmap_mode="r")
    s1_country, t_country = s1["country"].to_numpy(dtype=object), tg["country"].to_numpy(dtype=object)
    t0 = time.perf_counter()
    r = search(q_emb, t_emb, q, s1_country, t_country, cfg["blocking"]["k_max"])
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(r).to_parquet(b4_path.with_name(b4_path.name + ".partial"), index=False)
    b4_path.with_name(b4_path.name + ".partial").replace(b4_path)
    log(f"B4 [{split}]: {len(q):,} queries, search {time.perf_counter() - t0:.0f}s")
    t0 = time.perf_counter()
    rev = reverse_search(q_emb, t_emb, s1_country, t_country)
    np.savez(emb_dir(cfg) / f"{split}_reverse_{tag}.partial.npz", **rev)
    (emb_dir(cfg) / f"{split}_reverse_{tag}.partial.npz").replace(rev_path)
    log(f"reverse search [{split}]: {len(tg):,} targets in {time.perf_counter() - t0:.0f}s")


def reverse_search(s1_emb, t_emb, s1_country, t_country) -> dict[str, np.ndarray]:
    """Each target's best and second-best S1 cosine (and best S1 row) within its country.

    Covers every S1 record of the split, so the target-side competition features do not
    depend on which pairs blocking happened to retrieve.
    """
    n = len(t_country)
    best = np.full(n, -1.0, np.float32)
    second = np.full(n, -1.0, np.float32)
    best_s1 = np.full(n, -1, np.int64)
    for c in pd.unique(t_country):
        tr = np.flatnonzero(t_country == c)
        sr = np.flatnonzero(s1_country == c)
        if len(sr) == 0:
            continue
        idx, val = gpu_topk(np.asarray(t_emb[tr]), np.asarray(s1_emb[sr]), 2)  # 1024 x 1M scores ~ 2 GB VRAM
        best[tr], best_s1[tr] = val[:, 0], sr[idx[:, 0]]
        if idx.shape[1] > 1:
            second[tr] = val[:, 1]
    return {"best": best, "second": second, "best_s1": best_s1}


def run(cfg: dict, step: str, split: str) -> None:
    """Dispatch one step of the embedding blocker (B4)."""
    if step == "baseline":
        recall_eval(cfg, "base")
    elif step == "time":
        finetune(cfg, max_steps=200)
    elif step == "train":
        finetune(cfg)
    elif step == "eval":
        recall_eval(cfg, "ft", ft_dir(cfg))
    elif step == "choose":  # keep the fine-tuned model only if BOTH countries improve
        base = json.loads((emb_dir(cfg) / "recall_base.json").read_text())
        ft = json.loads((emb_dir(cfg) / "recall_ft.json").read_text())
        gains = {c: round(ft[c]["pair_recall"] - base[c]["pair_recall"], 4)
                 for c in base if isinstance(base[c], dict) and c != "overall"}
        tag = "ft" if all(g > 0 for g in gains.values()) else "base"
        reason = ("fine-tuned recall@k improved in every country" if tag == "ft"
                  else "fine-tuned recall@k did not improve in every country; using the un-tuned model")
        (emb_dir(cfg) / "chosen.json").write_text(json.dumps({"tag": tag, "gains": gains, "reason": reason}, indent=1))
        log(f"B4 model: {tag} ({reason}; recall gains {gains})")
    elif step == "index":  # the configured model (blocking.b4: "ft" or "base")
        tag = cfg["blocking"]["b4"]
        index(cfg, split, ft_dir(cfg) if tag == "ft" else None, tag)
    else:
        raise SystemExit(f"unknown embed step '{step}'")
