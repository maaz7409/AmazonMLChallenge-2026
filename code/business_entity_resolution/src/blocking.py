"""Candidate generation (blocking).

All blockers run inside country partitions discovered from the data: the data checks found no
labeled pair spanning two country labels, so an S1 record is only compared with targets
carrying the same country string, and a test-only country (France) simply becomes one
more partition. Country is never a model feature.

* B1: TF-IDF over char_wb 2-4 grams of name_ascii; top-K targets per S1 record.
* B2: TF-IDF over char 3-grams of "name_core address_core"; top-K per S1 record.
* B3: key blocking through inverted indexes, skipping blocks with more than
  `key_block_cap` targets: exact name_core; postal + rarest name token;
  house_no + first street word; the two rarest name tokens together.

Scale: with ~10M targets, exact char-gram cosine search is too slow because common grams
have posting lists of millions. Retrieval therefore (a) hashes grams into 2**22 buckets
(HashingVectorizer, run in parallel processes), (b) ignores grams whose document
frequency exceeds `max_df` of the split's documents, and (c) queries with each S1
record's `query_top_m` rarest grams. Vectors are L2-normalized over all their grams
first, so retrieval scores are partial cosines (lower bounds of the full cosine).

Output: artifacts/candidates/{split}/part-*.parquet with one row per (S1, target) pair of
the union, its per-blocker scores/ranks (rank 0 = not retrieved) and key flags. K is
applied downstream, so the recall table can compare several K from one run.
"""

import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from sparse_dot_topn import sp_matmul_topn

from .io import as_arrow, s1_roles, truth_pairs
from .normalize.normalize import load_normalized
from .utils import log, n_threads, n_workers

ANALYZERS = {"b1": {"analyzer": "char_wb", "ngram_range": (2, 4)},
             "b2": {"analyzer": "char", "ngram_range": (3, 3)},
             "b2w": {"analyzer": "word", "token_pattern": r"\S+", "ngram_range": (1, 1)}}
TFIDF_BLOCKERS = ["b1", "b2", "b2w", "b4"]  # b4 = arctic-embed search, same result format
KEY_NAMES = ["k_name", "k_postal", "k_house", "k_rare2"]
JOB_SIZE = 50_000  # texts per worker job
_STATE = {}        # per-worker vectorizer state, set by _init_worker


def workers(cfg: dict) -> int:
    """Number of CPU worker processes to use."""
    return n_workers(cfg)


def threads(cfg: dict) -> int:
    """Number of threads for multi-threaded native code (shares memory, unlike processes)."""
    return n_threads(cfg)


def blocker_text(df: pd.DataFrame, kind: str) -> pa.Array:
    """Text a TF-IDF blocker indexes for each record, as a compact Arrow string array."""
    if kind == "b1":
        return as_arrow(df["name_ascii"])
    return as_arrow((df["name_core"] + " " + df["address_core"]).str.strip())  # b2, b2w


# ---------------------------------------------------------------- TF-IDF workers
def _init_worker(kind: str, n_features: int, idf, keep, top_m: int) -> None:
    _STATE["hv"] = HashingVectorizer(n_features=n_features, alternate_sign=False, norm=None,
                                     lowercase=False, dtype=np.float32, **ANALYZERS[kind])
    _STATE.update(idf=idf, keep=keep, top_m=top_m)


def _df_job(texts: pa.Array) -> tuple[np.ndarray, np.ndarray]:
    """Document frequency of the hashed grams in a chunk of texts, as (bucket, count)."""
    X = _STATE["hv"].transform(texts.to_pylist())
    buckets, counts = np.unique(X.indices, return_counts=True)
    return buckets.astype(np.int32), counts.astype(np.int32)


def _tfidf_job(args: tuple[list[str], bool]) -> sparse.csr_matrix:
    """TF-IDF rows, L2-normalized over all grams, restricted to retrieval grams.

    Target rows keep every gram with document frequency <= max_df; query rows also keep
    only their `top_m` rarest such grams.
    """
    texts, is_query = args
    X = _STATE["hv"].transform(texts.to_pylist()).tocsr()
    idf, top_m = _STATE["idf"], _STATE["top_m"]
    rows = np.repeat(np.arange(X.shape[0]), np.diff(X.indptr))
    X.data *= idf[X.indices]
    norms = np.sqrt(np.bincount(rows, weights=X.data.astype(np.float64) ** 2,
                                minlength=X.shape[0]))
    X.data /= np.maximum(norms, 1e-12)[rows].astype(np.float32)
    keep = _STATE["keep"][X.indices]
    if is_query and top_m:
        priority = np.where(keep, -idf[X.indices], np.inf)  # rarest first, dropped grams last
        order = np.lexsort((priority, rows))
        rank = np.empty(len(order), dtype=np.int64)
        rank[order] = np.arange(len(order)) - X.indptr[rows[order]]
        keep &= rank < top_m
    X.data[~keep] = 0
    X.eliminate_zeros()
    return X


def _chunks(arr: pa.Array, size: int = JOB_SIZE) -> list[pa.Array]:
    """Compact copies of consecutive slices (a pickled slice would drag its parent buffer)."""
    return [pa.concat_arrays([arr.slice(i, size)]) for i in range(0, len(arr), size)]


def _vectorize(pool: ProcessPoolExecutor, texts: pa.Array, is_query: bool) -> sparse.csr_matrix:
    parts = list(pool.map(_tfidf_job, [(c, is_query) for c in _chunks(texts)]))
    return sparse.vstack(parts, format="csr")


def tfidf_retrieve(cfg: dict, kind: str, s1_text: pa.Array, t_text: pa.Array,
                   s1_country: np.ndarray, t_country: np.ndarray,
                   queries: np.ndarray) -> dict[str, np.ndarray]:
    """Top-`k_max` targets per query S1 row for one TF-IDF blocker, within countries.

    Returns flat arrays s1 (row), t (row), score (partial cosine) and rank (1 = best).
    """
    b = {**cfg["blocking"], **cfg["blocking"]["tfidf"][kind]}  # per-blocker max_df, query_top_m
    n_docs = len(s1_text) + len(t_text)
    t0 = time.perf_counter()
    df = np.zeros(b["n_features"], dtype=np.int64)
    with ProcessPoolExecutor(workers(cfg), initializer=_init_worker,
                             initargs=(kind, b["n_features"], None, None, 0)) as pool:
        for buckets, counts in pool.map(_df_job, _chunks(s1_text) + _chunks(t_text)):
            df[buckets] += counts
    idf = (np.log((1 + n_docs) / (1 + df)) + 1).astype(np.float32)
    keep = df <= b["max_df"] * n_docs
    log(f"{kind}: document frequencies over {n_docs:,} docs in {time.perf_counter() - t0:.0f}s; "
        f"{int((df > 0).sum()):,} grams seen, {int(((df > 0) & ~keep).sum()):,} above max_df")

    out = {"s1": [], "t": [], "score": [], "rank": []}
    with ProcessPoolExecutor(workers(cfg), initializer=_init_worker,
                             initargs=(kind, b["n_features"], idf, keep, b["query_top_m"])) as pool:
        for country in pd.unique(s1_country[queries]):
            q = queries[s1_country[queries] == country]
            t = np.flatnonzero(t_country == country)
            if len(t) == 0:
                continue
            t0 = time.perf_counter()
            B = _vectorize(pool, t_text.take(pa.array(t)), False).T.tocsr()
            A = _vectorize(pool, s1_text.take(pa.array(q)), True)
            t1 = time.perf_counter()
            for start in range(0, A.shape[0], b["chunk"]):
                C = sp_matmul_topn(A[start:start + b["chunk"]], B, top_n=b["k_max"],
                                   threshold=0.0, sort=True, n_threads=threads(cfg))
                counts = np.diff(C.indptr)
                out["s1"].append(q[start + np.repeat(np.arange(len(counts)), counts)].astype(np.int32))
                out["t"].append(t[C.indices].astype(np.int32))
                out["score"].append(C.data.astype(np.float32))
                out["rank"].append((np.arange(C.nnz) - np.repeat(C.indptr[:-1], counts) + 1)
                                   .astype(np.int16))
            log(f"{kind} [{country}]: {len(q):,} queries x {len(t):,} targets "
                f"(target nnz {B.nnz:,}); vectorize {t1 - t0:.0f}s, search {time.perf_counter() - t1:.0f}s")
            del A, B
    return {k: np.concatenate(v) for k, v in out.items()}


# ---------------------------------------------------------------- key blocking (B3)
def rare_name_tokens(cores: pa.Array) -> tuple[np.ndarray, np.ndarray, pa.Array]:
    """Rarest and second-rarest name_core token of every record (-1 when absent).

    Document frequency is counted over all records passed in (S1 + targets of a split).
    Returns (rarest token id, second-rarest token id, vocabulary as an Arrow array).
    """
    lists = pc.split_pattern(cores, " ")
    doc = pc.list_parent_indices(lists).to_numpy().astype(np.int64)
    toks = pc.list_flatten(lists)
    valid = pc.not_equal(toks, "").to_numpy(zero_copy_only=False)
    enc = toks.dictionary_encode()
    tid = enc.indices.to_numpy().astype(np.int64)[valid]
    doc = doc[valid]
    n_vocab = len(enc.dictionary)
    pairs = np.unique(doc * n_vocab + tid)  # distinct (doc, token)
    udoc, utid = pairs // n_vocab, pairs % n_vocab
    df = np.bincount(utid, minlength=n_vocab)
    order = np.lexsort((utid, df[utid], udoc))  # per doc: rarest first, ties by token id
    sdoc, stid = udoc[order], utid[order]
    first = np.r_[True, sdoc[1:] != sdoc[:-1]]
    second = np.r_[False, first[:-1] & ~first[1:]]
    r1 = np.full(len(cores), -1, dtype=np.int64)
    r2 = np.full(len(cores), -1, dtype=np.int64)
    r1[sdoc[first]] = stid[first]
    r2[sdoc[second]] = stid[second]
    return r1, r2, enc.dictionary.cast(pa.large_string())


# First alphabetic word (3+ letters) after the first number token of address_core:
# "1795 westchester drive high point nc" -> "westchester".
_STREET_WORD = r"^(?:[^ ]+ )*?[0-9]+[a-z]? (?:[^ ]+ )*?(?P<w>[a-z]{3,})(?: |$)"


def block_keys(df: pd.DataFrame, r1: np.ndarray, r2: np.ndarray, vocab: pa.Array) -> dict:
    """The four B3 keys of every record as Arrow strings (null when not defined).

    Keys are prefixed with the country string, which partitions the inverted indexes.
    Built with vectorized Arrow kernels: no per-record Python strings.
    """
    def col(name):
        a = as_arrow(df[name])
        return pc.if_else(pc.equal(a, ""), pa.scalar(None, pa.large_string()), a)

    def token(ids):
        return vocab.take(pa.array(np.maximum(ids, 0), mask=ids < 0))

    def join(*parts):  # null if any part is null
        return pc.binary_join_element_wise(*parts, pa.scalar("|", pa.large_string()))

    country = col("country")
    t1, t2 = token(r1), token(r2)
    word = pc.struct_field(pc.extract_regex(col("address_core"), _STREET_WORD), [0])
    return {"k_name": join(country, col("name_core")),
            "k_postal": join(country, col("postal"), t1),
            "k_house": join(country, col("house_no"), word),
            "k_rare2": join(country, pc.min_element_wise(t1, t2, skip_nulls=False),
                            pc.max_element_wise(t1, t2, skip_nulls=False))}


def key_block_pairs(s1_keys: pa.Array, t_keys: pa.Array, queries: np.ndarray,
                    cap: int) -> tuple[np.ndarray, np.ndarray]:
    """(S1 row, target row) pairs sharing a key, skipping blocks of more than `cap` targets."""
    enc = pa.concat_arrays([s1_keys.take(pa.array(queries)), t_keys]).dictionary_encode()
    codes = pc.fill_null(enc.indices, -1).to_numpy().astype(np.int64)
    qc, tc = codes[:len(queries)], codes[len(queries):]
    t_rows = np.flatnonzero(tc >= 0)
    order = np.argsort(tc[t_rows], kind="stable")
    t_sorted = t_rows[order]
    sizes = np.bincount(tc[t_rows], minlength=len(enc.dictionary))
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    size_q = np.where(qc >= 0, sizes[np.maximum(qc, 0)], 0)
    hit = np.flatnonzero((size_q > 0) & (size_q <= cap))
    cnt = size_q[hit]
    offsets = np.repeat(starts[qc[hit]] - (np.cumsum(cnt) - cnt), cnt) + np.arange(cnt.sum())
    return np.repeat(queries[hit], cnt).astype(np.int32), t_sorted[offsets].astype(np.int32)


# ---------------------------------------------------------------- union and reporting
def union_candidates(n_targets: int, tfidf: dict, keyed: dict) -> dict[str, np.ndarray]:
    """Union of all blockers, sorted by (S1, target), with per-blocker scores/ranks/flags."""
    def code(s, t):
        return s.astype(np.int64) * n_targets + t

    codes = np.unique(np.concatenate([code(r["s1"], r["t"]) for r in tfidf.values()]
                                     + [code(*keyed[k]) for k in KEY_NAMES]))
    out = {"s1": (codes // n_targets).astype(np.int32), "t": (codes % n_targets).astype(np.int32)}
    for name, r in tfidf.items():
        pos = np.searchsorted(codes, code(r["s1"], r["t"]))
        score = np.full(len(codes), np.nan, dtype=np.float32)
        rank = np.zeros(len(codes), dtype=np.int16)
        score[pos], rank[pos] = r["score"], r["rank"]
        out[f"{name}_score"], out[f"{name}_rank"] = score, rank
    for k in KEY_NAMES:
        flag = np.zeros(len(codes), dtype=np.int8)
        flag[np.searchsorted(codes, code(*keyed[k]))] = 1
        out[k] = flag
    return out


def in_candidate_set(cand: dict | pd.DataFrame, k) -> np.ndarray:
    """Mask of pairs kept at cutoff K: any TF-IDF blocker rank <= K, or a used B3 key match.

    `k` is one cutoff for every TF-IDF blocker or a {blocker: cutoff} dict; `keys` in the
    dict form (default: all) lists the B3 keys that admit pairs.
    """
    ks = k if isinstance(k, dict) else {b: k for b in TFIDF_BLOCKERS}
    mask = np.zeros(len(cand["s1"]), dtype=bool)
    for b in TFIDF_BLOCKERS:
        if f"{b}_rank" in cand and ks.get(b, 0) > 0:
            r = np.asarray(cand[f"{b}_rank"])
            mask |= (r > 0) & (r <= ks[b])
    for key in ks.get("keys", KEY_NAMES):
        mask |= np.asarray(cand[key]) > 0
    return mask


def coverage(cand_s1, cand_t, true_s1, true_t, eval_rows, n_s1, n_targets) -> dict:
    """Recall statistics of a candidate set on the S1 rows in `eval_rows`."""
    ev = np.zeros(n_s1, dtype=bool)
    ev[eval_rows] = True
    keep = ev[cand_s1]
    c_codes = np.sort(cand_s1[keep].astype(np.int64) * n_targets + cand_t[keep])
    tk = ev[true_s1]
    t_codes = true_s1[tk].astype(np.int64) * n_targets + true_t[tk]
    pos = np.minimum(np.searchsorted(c_codes, t_codes), max(len(c_codes) - 1, 0))
    hit = (c_codes[pos] == t_codes) if len(c_codes) else np.zeros(len(t_codes), bool)
    n_true = np.bincount(true_s1[tk], minlength=n_s1)[eval_rows]
    n_hit = np.bincount(true_s1[tk][hit], minlength=n_s1)[eval_rows]
    per_s1 = np.bincount(cand_s1[keep], minlength=n_s1)[eval_rows]
    r = np.divide(n_hit, n_true, out=np.zeros(len(n_true)), where=n_true > 0)
    f_ceiling = np.where(n_true == 0, 1.0, np.where(n_hit > 0, 1.25 * r / (0.25 + r + 1e-12), 0.0))
    has = n_true > 0
    return {
        "pair_recall": round(float(hit.mean()), 4),
        "entity_recall_ceiling": round(float((n_hit[has] == n_true[has]).mean()), 4),
        "f05_ceiling": round(float(f_ceiling.mean()), 4),
        "mean_cands": round(float(per_s1.mean()), 1),
        "p99_cands": int(np.percentile(per_s1, 99)),
        "max_cands": int(per_s1.max()),
        "no_cands_pct": round(100 * float((per_s1 == 0).mean()), 2),
        "reduction_ratio": float(1 - keep.sum() / (len(eval_rows) * n_targets)),
    }


def recall_report(cfg: dict, cand: dict, roles: np.ndarray, s1: pd.DataFrame,
                  targets: pd.DataFrame) -> dict:
    """Recall table on validation S1 rows for K in `report_ks`, plus per-blocker and per-country views."""
    true_s1, true_t = truth_pairs(cfg, s1["key"].to_numpy(), targets["key"].to_numpy())
    val = np.flatnonzero(roles == 2)
    n_s1, n_t = len(s1), len(targets)
    s, t = cand["s1"], cand["t"]
    report = {"union": [], "single_blocker": [], "final": {}, "per_country": []}
    # With union_prune, ranks beyond each blocker's final K were dropped before the union,
    # so the per-K rows would be misleading; only the final set and B3 keys are reported.
    for k in ([] if cfg["blocking"].get("union_prune") else cfg["blocking"]["report_ks"]):
        m = in_candidate_set(cand, k)
        report["union"].append({"K": k, **coverage(s[m], t[m], true_s1, true_t, val, n_s1, n_t)})
        for name in (b for b in TFIDF_BLOCKERS if f"{b}_rank" in cand):
            m1 = (cand[f"{name}_rank"] > 0) & (cand[f"{name}_rank"] <= k)
            report["single_blocker"].append({"blocker": f"{name}@{k}",
                                             **coverage(s[m1], t[m1], true_s1, true_t, val, n_s1, n_t)})
    for key in KEY_NAMES:
        m1 = cand[key] > 0
        report["single_blocker"].append({"blocker": key,
                                         **coverage(s[m1], t[m1], true_s1, true_t, val, n_s1, n_t)})
    final_k = cfg["blocking"]["k"]
    m = in_candidate_set(cand, final_k)
    report["final"] = {"K": final_k, **coverage(s[m], t[m], true_s1, true_t, val, n_s1, n_t)}
    country = s1["country"].to_numpy()
    for c in pd.unique(country[val]):
        rows = val[country[val] == c]
        report["per_country"].append({"country": c,
                                      **coverage(s[m], t[m], true_s1, true_t, rows, n_s1, n_t)})
    return report


def block_universe(cfg: dict, s1: pd.DataFrame, targets: pd.DataFrame,
                   queries: np.ndarray, b4: dict | None = None) -> dict[str, np.ndarray]:
    """Every configured blocker on an in-memory universe (indexes built on it alone).

    Same retrieval and union as the blocking stage, without caching; used for the stress
    set's mini-universe. `b4` holds embedding-search results when a B4 model is configured.
    Returns the union with per-blocker scores/ranks and key flags.
    """
    s1_country = s1["country"].to_numpy(dtype=object)
    t_country = targets["country"].to_numpy(dtype=object)
    results = {kind: tfidf_retrieve(cfg, kind, blocker_text(s1, kind), blocker_text(targets, kind),
                                    s1_country, t_country, queries)
               for kind in cfg["blocking"]["tfidf"]}
    if b4 is not None:
        results["b4"] = b4
    if cfg["blocking"].get("union_prune"):  # same cutoffs as the blocking stage
        ks = cfg["blocking"]["k"]
        results = {n: {c: v[r["rank"] <= ks.get(n, 0)] for c, v in r.items()} for n, r in results.items()}
    r1, r2, vocab = rare_name_tokens(pa.concat_arrays([as_arrow(s1["name_core"]),
                                                       as_arrow(targets["name_core"])]))
    n1 = len(s1)
    k_s1 = block_keys(s1, r1[:n1], r2[:n1], vocab)
    k_t = block_keys(targets, r1[n1:], r2[n1:], vocab)
    keyed = {k: key_block_pairs(k_s1[k], k_t[k], queries, cfg["blocking"]["key_block_cap"])
             for k in KEY_NAMES}
    return union_candidates(len(targets), results, keyed)


def candidates_dir(cfg: dict, split: str):
    return cfg["paths"]["artifacts_dir"] / "candidates" / split


def query_rows(cfg: dict, split: str, s1_keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(S1 roles, S1 rows that get candidates). Test: every S1. Train: the training sample
    and validation, or every S1 when split.block_all_train_s1 is set."""
    if split != "train":
        return np.ones(len(s1_keys), np.int8), np.arange(len(s1_keys), dtype=np.int64)
    roles = s1_roles(cfg, s1_keys)
    if cfg["split"].get("block_all_train_s1"):
        return roles, np.arange(len(s1_keys), dtype=np.int64)
    return roles, np.flatnonzero(roles > 0).astype(np.int64)


BLOCK_COLUMNS = ["key", "country", "name_ascii", "name_core", "address_core", "postal", "house_no"]


def lexical_results(cfg: dict, split: str, s1: pd.DataFrame, targets: pd.DataFrame,
                    queries: np.ndarray, kinds=None, tag: str = "") -> dict:
    """Cached top-`k_max` results of the configured TF-IDF blockers for the given query S1
    rows (all of them, or `kinds`). `tag` names a query subset (e.g. the embedder's S1)."""
    b = cfg["blocking"]
    out_dir = candidates_dir(cfg, split)
    out_dir.mkdir(parents=True, exist_ok=True)
    s1_country = s1["country"].to_numpy(dtype=object)
    t_country = targets["country"].to_numpy(dtype=object)
    results = {}
    for kind in (kinds or list(b["tfidf"])):
        st = b["tfidf"][kind]
        path = out_dir / f"_{kind}{tag}_q{len(queries)}_df{st['max_df']}_m{st['query_top_m']}_k{b['k_max']}.parquet"
        if path.exists():
            results[kind] = {c: v.to_numpy() for c, v in pd.read_parquet(path).items()}
            continue
        t0 = time.perf_counter()
        results[kind] = tfidf_retrieve(cfg, kind, blocker_text(s1, kind), blocker_text(targets, kind),
                                       s1_country, t_country, queries)
        tmp = path.with_name(path.name + ".partial")
        pd.DataFrame(results[kind]).to_parquet(tmp, index=False)
        tmp.replace(path)
        log(f"{kind}{tag} [{split}]: {len(queries):,} queries in {time.perf_counter() - t0:.0f}s")
    return results


def run_lexical(cfg: dict, split: str) -> None:
    """Blocking step 1 (CPU): the TF-IDF blockers for every query S1 of a split (cached)."""
    s1 = load_normalized(cfg, split, "s1", columns=BLOCK_COLUMNS)
    targets = load_normalized(cfg, split, "targets", columns=BLOCK_COLUMNS)
    _, queries = query_rows(cfg, split, s1["key"].to_numpy())
    lexical_results(cfg, split, s1, targets, queries)


def lexical_negatives(cfg: dict, s1_rows: np.ndarray) -> pd.DataFrame:
    """(s1, t, b2w_rank, b1_rank) lexical top candidates of the given train S1 rows, from
    the full lexical cache when it exists, else from a cached run on just these rows (so
    the embedder can start training long before the full lexical blocking has finished)."""
    s1 = load_normalized(cfg, "train", "s1", columns=BLOCK_COLUMNS)
    targets = load_normalized(cfg, "train", "targets", columns=BLOCK_COLUMNS)
    _, queries = query_rows(cfg, "train", s1["key"].to_numpy())
    b = cfg["blocking"]
    full = all((candidates_dir(cfg, "train") / f"_{k}_q{len(queries)}_df{st['max_df']}_m{st['query_top_m']}"
                f"_k{b['k_max']}.parquet").exists() for k, st in b["tfidf"].items())
    if full:
        res = lexical_results(cfg, "train", s1, targets, queries)
    else:
        res = lexical_results(cfg, "train", s1, targets, np.sort(s1_rows).astype(np.int64), tag="_emb")
    frames = []
    keep = np.zeros(len(s1), bool)
    keep[s1_rows] = True
    for kind, r in res.items():
        m = keep[r["s1"]] & (r["rank"] <= 10)
        frames.append(pd.DataFrame({"s1": r["s1"][m], "t": r["t"][m], f"{kind}_rank": r["rank"][m]}))
    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on=["s1", "t"], how="outer")
    for kind in res:
        out[f"{kind}_rank"] = out[f"{kind}_rank"].fillna(0).astype(np.int16)
    return out


def run(cfg: dict, split: str) -> None:
    """Blocking step 2: union of the lexical blockers and B4 at the pool cutoffs
    (`blocking.k`), saved as candidate parts (plus the recall table on train)."""
    t_start = time.perf_counter()
    s1 = load_normalized(cfg, split, "s1", columns=BLOCK_COLUMNS)
    targets = load_normalized(cfg, split, "targets", columns=BLOCK_COLUMNS)
    roles, queries = query_rows(cfg, split, s1["key"].to_numpy())
    log(f"blocking {split}: {len(queries):,} query S1 of {len(s1):,}; {len(targets):,} targets")
    s1_country = s1["country"].to_numpy(dtype=object)
    out_dir = candidates_dir(cfg, split)
    out_dir.mkdir(parents=True, exist_ok=True)
    b = cfg["blocking"]
    ks = b["k"] if isinstance(b["k"], dict) else None

    # skip when the union is up to date with its inputs (lexical and B4 caches, settings)
    inputs = sorted(p.name + f"@{p.stat().st_mtime_ns}" for p in out_dir.glob("_*.parquet"))
    stamp = {"settings": json.loads(json.dumps(b, default=str)), "inputs": inputs, "queries": len(queries)}
    stamp_path = out_dir / "union_stamp.json"
    if stamp_path.exists() and json.loads(stamp_path.read_text()) == stamp and any(out_dir.glob("part-*.parquet")):
        log(f"union [{split}] already built; skipped")
        return
    stamp_path.unlink(missing_ok=True)

    t0 = time.perf_counter()
    results = lexical_results(cfg, split, s1, targets, queries)
    timings = {"lexical": round(time.perf_counter() - t0, 1)}

    use_keys = KEY_NAMES if ks is None else ks.get("keys", KEY_NAMES)
    if use_keys:
        def b3_pairs() -> dict:
            r1, r2, vocab = rare_name_tokens(pa.concat_arrays([as_arrow(s1["name_core"]),
                                                               as_arrow(targets["name_core"])]))
            n1 = len(s1)
            k_s1 = block_keys(s1, r1[:n1], r2[:n1], vocab)
            k_t = block_keys(targets, r1[n1:], r2[n1:], vocab)
            pairs = [key_block_pairs(k_s1[k], k_t[k], queries, b["key_block_cap"]) for k in KEY_NAMES]
            return {"s1": np.concatenate([p[0] for p in pairs]), "t": np.concatenate([p[1] for p in pairs]),
                    "key": np.repeat(np.arange(len(KEY_NAMES), dtype=np.int8), [len(p[0]) for p in pairs])}

        t0 = time.perf_counter()
        path = out_dir / f"_b3_q{len(queries)}_cap{b['key_block_cap']}.parquet"
        if path.exists():
            b3 = {c: v.to_numpy() for c, v in pd.read_parquet(path).items()}
        else:
            b3 = b3_pairs()
            pd.DataFrame(b3).to_parquet(path, index=False)
        keyed = {k: (b3["s1"][b3["key"] == i], b3["t"][b3["key"] == i]) for i, k in enumerate(KEY_NAMES)}
        keyed = {k: (v if k in use_keys else (v[0][:0], v[1][:0])) for k, v in keyed.items()}
        timings["b3"] = round(time.perf_counter() - t0, 1)
        del b3
    else:  # the pool uses no key blocks: skip them entirely
        empty = np.zeros(0, np.int32)
        keyed = {k: (empty, empty) for k in KEY_NAMES}
    log("b3 pairs: " + ", ".join(f"{k} {len(v[0]):,}" for k, v in keyed.items()))

    if b.get("b4"):  # embedding blocker results from `--stage embed --step index`
        results["b4"] = {c: v.to_numpy() for c, v in pd.read_parquet(
            out_dir / f"_b4_{b['b4']}_q{len(queries)}_k{b['k_max']}.parquet").items()}
    if b.get("union_prune"):  # keep only pairs inside the per-blocker cutoffs
        kd = ks if ks is not None else {name: b["k"] for name in results}
        for name, r in results.items():
            keep = r["rank"] <= kd.get(name, 0)
            results[name] = {c: v[keep] for c, v in r.items()}

    # Union per country (S1 records never span countries), so peak memory stays bounded;
    # every part holds complete S1 groups. Validation rows are kept for the recall table.
    for old in list(out_dir.glob("part-*.parquet")) + list(out_dir.glob("*.partial")):
        old.unlink()
    part_no, n_pairs, val_cands = 0, 0, []
    for c in pd.unique(s1_country[queries]):
        in_c = np.zeros(len(s1), dtype=bool)
        in_c[queries[s1_country[queries] == c]] = True
        cand = union_candidates(len(targets),
                                {n: {k: v[in_c[r["s1"]]] for k, v in r.items()} for n, r in results.items()},
                                {k: (p[0][in_c[p[0]]], p[1][in_c[p[0]]]) for k, p in keyed.items()})
        n_pairs += len(cand["s1"])
        log(f"union [{c}]: {len(cand['s1']):,} candidate pairs")
        change = np.r_[True, cand["s1"][1:] != cand["s1"][:-1]]
        part = (np.cumsum(change) - 1) // b["s1_per_part"]
        bounds = np.searchsorted(part, np.arange(part.max() + 2))
        for i in range(len(bounds) - 1):
            sl = slice(bounds[i], bounds[i + 1])
            pd.DataFrame({k: v[sl] for k, v in cand.items()}).to_parquet(
                out_dir / f"part-{part_no:04d}.parquet", index=False)
            part_no += 1
        if split == "train":
            v = roles[cand["s1"]] == 2
            val_cands.append({k: arr[v] for k, arr in cand.items()})
        del cand
    del results, keyed

    meta = {"split": split, "queries": len(queries), "pairs": n_pairs,
            "timings_s": timings, "total_s": round(time.perf_counter() - t_start, 1),
            "settings": cfg["blocking"]}
    if split == "train":
        cand = {k: np.concatenate([vc[k] for vc in val_cands]) for k in val_cands[0]}
        meta["recall"] = recall_report(cfg, cand, roles, s1, targets)
    with open(out_dir / "blocking_report.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)
    if "recall" in meta:
        print(json.dumps(meta["recall"], indent=1))
    stamp["inputs"] = sorted(p.name + f"@{p.stat().st_mtime_ns}" for p in out_dir.glob("_*.parquet"))
    stamp_path.write_text(json.dumps(stamp))
    log(f"blocking {split} done in {meta['total_s']}s")
