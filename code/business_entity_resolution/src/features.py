"""Pair and group features for candidate (S1, target) pairs.

Features are computed per candidates part (each part holds complete S1 groups), fully
vectorized: string similarities through rapidfuzz.process.cpdist (multi-threaded, one
call per scorer and view), token overlaps through sparse row gathers, and within-S1
group features through sort / reduceat. Country is never used.

Output: artifacts/features/{split}/part-*.parquet with s1, t, label (train only) and
float32 feature columns.
"""

import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein
from scipy import sparse

from .blocking import KEY_NAMES, TFIDF_BLOCKERS, candidates_dir, in_candidate_set, workers
from .io import KEY_BASE, as_arrow, s1_roles, truth_pairs
from .normalize.normalize import load_normalized
from .utils import fork_map, log, n_procs, n_threads

STRING_VIEWS = ["name_ascii", "name_core", "legal_form", "address_core", "landmark",
                "house_no", "postal"]
NAME_SCORERS = {"ratio": fuzz.ratio, "tset": fuzz.token_set_ratio, "tsort": fuzz.token_sort_ratio,
                "partial": fuzz.partial_ratio, "jw": JaroWinkler.normalized_similarity,
                "lev": Levenshtein.normalized_similarity}
ADDRESS_SCORERS = {"tset": fuzz.token_set_ratio, "tsort": fuzz.token_sort_ratio,
                   "ratio": fuzz.ratio, "jw": JaroWinkler.normalized_similarity}
GROUP_SOURCES = ["name_core_tset", "name_core_jw", "addr_tset", "name_idf_jacc",
                 "addr_idf_jacc", "combo"]
PAIRS_PER_CHUNK = 1_000_000
# Bump whenever feature definitions change: cached feature parts of another version are recomputed.
FEATURES_VERSION = "4.0-pruned-prune-p"
RARE_IDF = 8.0  # tokens with idf above this (document frequency below ~0.03%) count as rare


class TokenIndex:
    """Binary token matrices of S1 and targets for one text view, with IDF weights.

    Tokens are whitespace-separated words of the view. Global IDF counts document
    frequency over all S1 + target records of the split; when record countries are given,
    a second, per-country IDF is kept too (self-adapting to each country's vocabulary,
    France included), exposed as "<prefix>_pc_*" features.
    """

    def __init__(self, s1_values, t_values, s1_country=None, t_country=None):
        arr = pa.concat_arrays([as_arrow(s1_values), as_arrow(t_values)])
        lists = pc.split_pattern(arr, " ")
        flat = pc.list_flatten(lists)
        valid = pc.not_equal(flat, "")
        rows = pc.list_parent_indices(lists).filter(valid).to_numpy()  # sorted by row
        enc = flat.filter(valid).dictionary_encode()
        n, vocab = len(arr), len(enc.dictionary)
        indptr = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(np.bincount(rows, minlength=n), out=indptr[1:])
        cols = enc.indices.to_numpy().astype(np.int32)
        X = sparse.csr_matrix((np.ones(len(cols), np.float32), cols, indptr), shape=(n, vocab))
        del rows, cols
        X.sum_duplicates()  # a token repeated in one record counts once
        X.data[:] = 1.0
        df = np.bincount(X.indices, minlength=vocab)
        self.vocab = vocab
        self.idf = np.log(n / np.maximum(df, 1)).astype(np.float32)
        n1 = len(s1_values)
        self.s1, self.t = X[:n1], X[n1:]
        self.s1_count = np.diff(self.s1.indptr)
        self.t_count = np.diff(self.t.indptr)
        self.s1_mass = self._mass(self.s1, None)
        self.t_mass = self._mass(self.t, None)
        self.idf_pc = None
        if s1_country is not None:
            codes, _ = pd.factorize(np.concatenate([np.asarray(s1_country, dtype=object),
                                                    np.asarray(t_country, dtype=object)]))
            self.cc_s1, self.cc_t = codes[:n1].astype(np.int64), codes[n1:].astype(np.int64)
            n_c = codes.max() + 1
            row_c = np.repeat(codes.astype(np.int64), np.diff(X.indptr))
            df_pc = np.bincount(row_c * vocab + X.indices, minlength=n_c * vocab).reshape(n_c, vocab)
            n_docs = np.bincount(codes, minlength=n_c)[:, None]
            self.idf_pc = np.log(n_docs / np.maximum(df_pc, 1)).astype(np.float32).ravel()
            self.s1_mass_pc = self._mass(self.s1, self.cc_s1)
            self.t_mass_pc = self._mass(self.t, self.cc_t)

    def _weights(self, indices: np.ndarray, row_country) -> np.ndarray:
        """IDF of each stored token: global, or per country of the row it belongs to."""
        return self.idf[indices] if row_country is None else self.idf_pc[row_country * self.vocab + indices]

    def _mass(self, X, record_country) -> np.ndarray:
        """Sum of token IDF per record."""
        rc = None if record_country is None else np.repeat(record_country, np.diff(X.indptr))
        return np.add.reduceat(np.r_[self._weights(X.indices, rc), 0.0], X.indptr[:-1]) \
            * (np.diff(X.indptr) > 0)

    def jaccard(self, s: np.ndarray, t: np.ndarray) -> np.ndarray:
        """IDF-weighted Jaccard of pairs (S1 row s[i], target row t[i]) with the global IDF
        (the cheap subset of overlap() used by the candidate pruner)."""
        shared = self.s1[s].multiply(self.t[t]).tocsr()
        sm = np.add.reduceat(np.r_[self.idf[shared.indices], 0.0], shared.indptr[:-1]) \
            * (np.diff(shared.indptr) > 0)
        return (sm / np.maximum(self.s1_mass[s] + self.t_mass[t] - sm, 1e-6)).astype(np.float32)

    @staticmethod
    def _row_max(M) -> np.ndarray:
        out = np.zeros(M.shape[0], np.float32)
        nz = np.diff(M.indptr) > 0
        if nz.any():
            out[nz] = np.maximum.reduceat(M.data, M.indptr[:-1][nz])
        return out

    def overlap(self, s: np.ndarray, t: np.ndarray, prefix: str) -> dict[str, np.ndarray]:
        """IDF-weighted overlap features of pairs (S1 row s[i], target row t[i]).

        Shared-token evidence (Fellegi-Sunter style agreement weights) plus the rarest
        token each side has that the other lacks (strong disagreement evidence).
        """
        A, B = self.s1[s], self.t[t]
        shared = A.multiply(B).tocsr()
        only_a = (A - shared).tocsr()
        only_b = (B - shared).tocsr()
        only_a.eliminate_zeros()
        only_b.eliminate_zeros()
        out = {f"{prefix}_n_shared": np.diff(shared.indptr).astype(np.float32),
               f"{prefix}_token_diff": np.abs(self.s1_count[s] - self.t_count[t]).astype(np.float32)}
        variants = [(prefix, None, self.s1_mass[s], self.t_mass[t])]
        if self.idf_pc is not None:
            variants.append((f"{prefix}_pc", self.cc_s1[s], self.s1_mass_pc[s], self.t_mass_pc[t]))
        for name, pair_c, a, b in variants:
            def weighted(M):
                rc = None if pair_c is None else np.repeat(pair_c, np.diff(M.indptr))
                return sparse.csr_matrix((self._weights(M.indices, rc), M.indices, M.indptr), shape=M.shape)
            sh = weighted(shared)
            sm = np.asarray(sh.sum(axis=1)).ravel()
            union = np.maximum(a + b - sm, 1e-6)
            out.update({f"{name}_idf_jacc": sm / union,
                        f"{name}_idf_cover_s1": sm / np.maximum(a, 1e-6),
                        f"{name}_idf_cover_t": sm / np.maximum(b, 1e-6),
                        f"{name}_max_shared_idf": self._row_max(sh),
                        f"{name}_n_shared_rare": np.asarray((sh > RARE_IDF).sum(axis=1)).ravel(),
                        f"{name}_max_unmatched_idf_s1": self._row_max(weighted(only_a)),
                        f"{name}_max_unmatched_idf_t": self._row_max(weighted(only_b))})
        return out


def name_frequency(s1: pd.DataFrame, tg: pd.DataFrame) -> dict[str, np.ndarray]:
    """Per-country counts of records sharing each exact name_core ("chain" evidence).

    About half of S1 records share their name_core with other businesses, so a perfect
    name match is weaker evidence the more records carry that name.
    """
    key = pa.concat_arrays([as_arrow(s1["country"] + "|" + s1["name_core"]),
                            as_arrow(tg["country"] + "|" + tg["name_core"])]).dictionary_encode()
    codes = key.indices.to_numpy().astype(np.int64)
    n1 = len(s1)
    in_s1 = np.bincount(codes[:n1], minlength=len(key.dictionary))
    in_t = np.bincount(codes[n1:], minlength=len(key.dictionary))
    return {"s1_in_s1": in_s1[codes[:n1]], "s1_in_t": in_t[codes[:n1]],
            "t_in_s1": in_s1[codes[n1:]], "t_in_t": in_t[codes[n1:]]}


RF_WORKERS = {"n": -1}  # rapidfuzz threads; forked part workers set their share


def _cpdist(a, b, scorer) -> np.ndarray:
    return process.cpdist(a, b, scorer=scorer, processor=None, workers=RF_WORKERS["n"], dtype=np.float32)


def _eq_state(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """1 = equal, 0 = both present and different, NaN = either missing."""
    present = (a != "") & (b != "")
    return np.where(present, (a == b).astype(np.float32), np.nan).astype(np.float32)


def string_table(df: pd.DataFrame) -> pa.Table:
    """Arrow table of the string views used by pair features, plus acronym and compact name.

    acronym = initials of a multi-word name_core ("bharat heavy electricals" -> "bhe", else
    ""); compact = name_core without spaces. Both are built with vectorized Arrow kernels.
    """
    core = as_arrow(df["name_core"])
    words = pc.split_pattern(core, " ")
    initials = pa.LargeListArray.from_arrays(
        pc.cast(words.offsets, pa.int64()),
        pc.utf8_slice_codeunits(pc.list_flatten(words), 0, 1).cast(pa.large_string()))
    acr = pc.if_else(pc.greater_equal(pc.list_value_length(words), 2),
                     pc.binary_join(initials, pa.scalar("", pa.large_string())),
                     pa.scalar("", pa.large_string()))
    cols = {c: as_arrow(df[c]) for c in STRING_VIEWS}
    cols.update(acronym=acr, compact=pc.replace_substring(core, " ", ""))
    return pa.table(cols)


def pair_features(s: np.ndarray, t: np.ndarray, S1: pa.Table, T: pa.Table, tok: dict,
                  t_keys: np.ndarray, lengths: dict, freq: dict) -> dict[str, np.ndarray]:
    """All pair-level features for aligned index arrays s (S1 rows) and t (target rows)."""
    ss, tt = pa.array(s), pa.array(t)
    a = {c: S1[c].take(ss).to_numpy(zero_copy_only=False) for c in S1.column_names}
    b = {c: T[c].take(tt).to_numpy(zero_copy_only=False) for c in T.column_names}
    f = {}
    for view in ("name_core", "name_ascii"):
        for name, scorer in NAME_SCORERS.items():
            f[f"{view}_{name}"] = _cpdist(a[view], b[view], scorer)
    a_empty, b_empty = a["address_core"] == "", b["address_core"] == ""
    for name, scorer in ADDRESS_SCORERS.items():
        v = _cpdist(a["address_core"], b["address_core"], scorer)
        v[a_empty | b_empty] = np.nan
        f[f"addr_{name}"] = v
    lm = _cpdist(a["landmark"], b["landmark"], fuzz.token_set_ratio)
    lm[(a["landmark"] == "") | (b["landmark"] == "")] = np.nan
    f["landmark_tset"] = lm
    f.update(tok["name"].overlap(s, t, "name"))
    f.update(tok["addr"].overlap(s, t, "addr"))
    f.update(tok["num"].overlap(s, t, "num"))
    f["acronym_match"] = (((a["acronym"] != "") & (a["acronym"] == b["compact"])) |
                          ((b["acronym"] != "") & (b["acronym"] == a["compact"]))).astype(np.float32)
    f["postal_eq"] = _eq_state(a["postal"], b["postal"])
    f["house_eq"] = _eq_state(a["house_no"], b["house_no"])
    f["legal_eq"] = _eq_state(a["legal_form"], b["legal_form"])
    la, lb = lengths["s1_name"][s], lengths["t_name"][t]
    f["name_len_ratio"] = np.minimum(la, lb) / np.maximum(np.maximum(la, lb), 1)
    aa, ab = lengths["s1_addr"][s], lengths["t_addr"][t]
    f["addr_len_ratio"] = np.where(a_empty | b_empty, np.nan, np.minimum(aa, ab) / np.maximum(np.maximum(aa, ab), 1))
    f["t_addr_empty"] = b_empty.astype(np.float32)
    f["t_landmark"] = (b["landmark"] != "").astype(np.float32)
    f["s1_landmark"] = (a["landmark"] != "").astype(np.float32)
    f["t_is_s2"] = (t_keys[t] // KEY_BASE == 2).astype(np.float32)
    f["combo"] = (f["name_core_tset"] + np.nan_to_num(f["addr_tset"], nan=0.0)) / 2
    f["name_freq_s1_in_s1"] = np.log1p(freq["s1_in_s1"][s]).astype(np.float32)
    f["name_freq_s1_in_t"] = np.log1p(freq["s1_in_t"][s]).astype(np.float32)
    f["name_freq_t_in_s1"] = np.log1p(freq["t_in_s1"][t]).astype(np.float32)
    f["name_freq_t_in_t"] = np.log1p(freq["t_in_t"][t]).astype(np.float32)
    return f


def group_features(s: np.ndarray, f: dict, blockers: list[str]) -> dict[str, np.ndarray]:
    """Within-S1 context: candidate count, and rank / gap (and margin) per score."""
    starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
    sizes = np.diff(np.r_[starts, len(s)])
    gid = np.repeat(np.arange(len(starts)), sizes)
    out = {"n_cands": sizes[gid].astype(np.float32),
           "n_cands_s2": np.add.reduceat(f["t_is_s2"], starts)[gid]}
    extra = [n for n in ("emb_cos", "prune_p") if n in f]
    for name in GROUP_SOURCES + [f"{b}_score" for b in blockers] + extra:
        missing = np.isnan(f[name])
        v = np.where(missing, -1.0, f[name]).astype(np.float32)  # missing ranks last
        order = np.lexsort((-v, gid))
        rank = np.empty(len(v), np.float32)
        rank[order] = np.arange(len(v)) - starts[gid[order]] + 1
        best = np.maximum.reduceat(v, starts)[gid]
        out[f"{name}_grank"] = np.where(missing, np.nan, rank)
        out[f"{name}_gap"] = np.where(missing, np.nan, best - v)
        if name in GROUP_SOURCES:
            second = np.full(len(starts), -1.0, np.float32)
            multi = sizes > 1
            sv = v[order]
            second[multi] = sv[starts[multi] + 1]
            out[f"{name}_margin"] = np.where(missing, np.nan, v - np.where(rank == 1, second[gid], best))
    return out


def features_dir(cfg: dict, split: str):
    """Feature parts of a split; a named candidate set (blocking.set, e.g. "compact" for
    tighter cutoffs over the same candidate parts) gets its own directory."""
    name = cfg["blocking"].get("set")
    return cfg["paths"]["artifacts_dir"] / "features" / (f"{split}_{name}" if name else split)


def load_embeddings(cfg: dict, split: str) -> dict | None:
    """Embeddings of the configured B4 model (blocking.b4) for a split, memory-mapped.

    Includes the reverse-search arrays (best / second-best S1 cosine per target) when the
    embed index step produced them; None when no B4 model is configured.
    """
    tag = cfg["blocking"].get("b4")
    if not tag:
        return None
    d = cfg["paths"]["artifacts_dir"] / "embeddings"
    emb = {"s1": np.load(d / f"{split}_s1_{tag}.npy", mmap_mode="r"),
           "t": np.load(d / f"{split}_targets_{tag}.npy", mmap_mode="r")}
    rev = d / f"{split}_reverse_{tag}.npz"
    if rev.exists():
        with np.load(rev) as z:
            emb.update(rev_best=z["best"], rev_second=z["second"], rev_best_s1=z["best_s1"])
    return emb


FEATURE_COLUMNS = ["key", "country"] + STRING_VIEWS + ["numbers"]  # normalized columns needed


def target_competition(frames, n_targets: int) -> dict[str, np.ndarray]:
    """Per target, over every S1 record whose final candidate set contains it:
    the best and second-best B2w score, the best S1 row and the number of such S1.

    `frames` yields candidate DataFrames (s1, t, b2w_score); on train this needs every S1
    blocked (split.block_all_train_s1) so competition matches test, where all are blocked.
    """
    best = np.full(n_targets, -1.0, np.float32)
    second = np.full(n_targets, -1.0, np.float32)
    best_s1 = np.full(n_targets, -1, np.int64)
    count = np.zeros(n_targets, np.int32)
    for c in frames:
        t = c["t"].to_numpy()
        sc = np.nan_to_num(c["b2w_score"].to_numpy(np.float32), nan=-1.0) if "b2w_score" in c \
            else np.full(len(c), -1.0, np.float32)
        s1 = c["s1"].to_numpy()
        count += np.bincount(t, minlength=n_targets).astype(np.int32)
        order = np.lexsort((-sc, t))
        ts, ss, s1s = t[order], sc[order], s1[order]
        first = np.r_[True, ts[1:] != ts[:-1]]
        idx = np.flatnonzero(first)
        tg, b1, bs1 = ts[idx], ss[idx], s1s[idx]
        nxt = np.minimum(idx + 1, len(ts) - 1)
        b2 = np.where((nxt > idx) & (ts[nxt] == tg), ss[nxt], -1.0).astype(np.float32)
        old_b, old_s = best[tg], second[tg]
        new_best = b1 > old_b
        second[tg] = np.where(new_best, np.maximum(old_b, b2), np.maximum(old_s, b1))
        best[tg] = np.where(new_best, b1, old_b)
        best_s1[tg] = np.where(new_best, bs1, best_s1[tg])
    return {"best": best, "second": second, "best_s1": best_s1, "count": count}


class FeatureContext:
    """Everything pair features need about one universe of S1 and target records.

    Built once per universe (a split, or the stress set's mini-universe): string tables,
    token indexes with IDF from that universe, name frequencies and field lengths.
    """

    def __init__(self, s1: pd.DataFrame, tg: pd.DataFrame, competition: dict | None = None,
                 emb: dict | None = None):
        """competition: target_competition() output; emb: {"s1", "t"} embedding arrays and
        optionally {"rev_best", "rev_second", "rev_best_s1"} from the reverse search."""
        self.competition, self.emb = competition, emb
        self.n_targets = len(tg)
        self.S1, self.T = string_table(s1), string_table(tg)
        s1_c, t_c = s1["country"].to_numpy(dtype=object), tg["country"].to_numpy(dtype=object)
        self.tok = {"name": TokenIndex(s1["name_core"], tg["name_core"], s1_c, t_c),
                    "addr": TokenIndex(s1["address_core"], tg["address_core"], s1_c, t_c),
                    "num": TokenIndex(s1["numbers"], tg["numbers"])}
        self.freq = name_frequency(s1, tg)
        self.t_keys = tg["key"].to_numpy()
        self.lengths = {"s1_name": s1["name_core"].str.len().to_numpy(np.float32),
                        "t_name": tg["name_core"].str.len().to_numpy(np.float32),
                        "s1_addr": s1["address_core"].str.len().to_numpy(np.float32),
                        "t_addr": tg["address_core"].str.len().to_numpy(np.float32)}

    def compute(self, cand: pd.DataFrame) -> pd.DataFrame:
        """Features of candidate pairs (rows sorted by s1, complete S1 groups): s1, t, features."""
        s, t = cand["s1"].to_numpy(), cand["t"].to_numpy()
        feats = {}
        for lo in range(0, len(s), PAIRS_PER_CHUNK):  # group features below use whole S1 groups
            hi = min(lo + PAIRS_PER_CHUNK, len(s))
            for name, v in pair_features(s[lo:hi], t[lo:hi], self.S1, self.T, self.tok, self.t_keys,
                                         self.lengths, self.freq).items():
                feats.setdefault(name, []).append(v)
        feats = {name: np.concatenate(v).astype(np.float32) for name, v in feats.items()}
        blockers = [b for b in TFIDF_BLOCKERS if f"{b}_rank" in cand.columns]
        for b in blockers:
            feats[f"{b}_score"] = cand[f"{b}_score"].to_numpy(np.float32)
            rank = cand[f"{b}_rank"].to_numpy().astype(np.float32)
            rank[rank == 0] = np.nan  # not retrieved by this blocker
            feats[f"{b}_rank"] = rank
        for key in KEY_NAMES:
            feats[key] = cand[key].to_numpy(np.float32)
        if "prune_p" in cand.columns:  # the candidate pruner's probability (out-of-sample)
            feats["prune_p"] = cand["prune_p"].to_numpy(np.float32)
        if self.competition is not None:  # does another S1 claim this target more strongly?
            c = self.competition
            own = np.nan_to_num(feats.get("b2w_score", np.full(len(s), np.nan)), nan=-1.0)
            is_best = c["best_s1"][t] == s
            other = np.where(is_best, c["second"][t], c["best"][t])
            feats["t_n_s1_cands"] = c["count"][t].astype(np.float32)
            feats["b2w_t_best_other_s1"] = other.astype(np.float32)
            feats["b2w_margin_vs_other_s1"] = (own - other).astype(np.float32)
            feats["b2w_is_t_argmax"] = is_best.astype(np.float32)
        if self.emb is not None:  # fine-tuned arctic-embed cosine and its competition
            # chunked: gathering both 256-d vectors for a whole part at once takes ~7 GB
            cos = np.empty(len(s), np.float32)
            for a in range(0, len(s), 262_144):
                b = a + 262_144
                cos[a:b] = np.einsum("ij,ij->i", np.asarray(self.emb["s1"][s[a:b]], np.float32),
                                     np.asarray(self.emb["t"][t[a:b]], np.float32))
            feats["emb_cos"] = cos
            if "rev_best" in self.emb:
                is_best = self.emb["rev_best_s1"][t] == s
                other = np.where(is_best, self.emb["rev_second"][t], self.emb["rev_best"][t])
                feats["emb_t_best_other_s1_cos"] = other.astype(np.float32)
                feats["emb_margin_vs_other_s1"] = (cos - other).astype(np.float32)
                feats["emb_is_t_argmax"] = is_best.astype(np.float32)
        feats.update(group_features(s, feats, blockers))
        return pd.DataFrame({"s1": s, "t": t, **feats})


def final_candidates_dir(cfg: dict, split: str):
    """Candidate parts the matching models score: the pruned set (`prune.enabled`) or
    the blocking union at the `blocking.k` cutoffs."""
    if cfg.get("prune", {}).get("enabled"):
        from .prune import pruned_dir
        return pruned_dir(cfg, split)
    return candidates_dir(cfg, split)


def _feature_part(ctx: "FeatureContext", job) -> int:
    """Worker: features of one candidate part, written atomically; returns its pair count."""
    path, dest = job
    cand = pd.read_parquet(path)
    if ctx.k is not None:
        cand = cand[in_candidate_set(cand, ctx.k)].reset_index(drop=True)
    out = ctx.compute(cand)
    s, t = out["s1"].to_numpy(), out["t"].to_numpy()
    if ctx.label_codes is not None:
        codes = s.astype(np.int64) * ctx.n_targets + t
        pos = np.minimum(np.searchsorted(ctx.label_codes, codes), len(ctx.label_codes) - 1)
        out["label"] = (ctx.label_codes[pos] == codes).astype(np.int8)
        out["role"] = ctx.roles[s]
    tmp = dest.with_name(dest.name + ".partial")
    out.to_parquet(tmp, index=False)
    tmp.replace(dest)  # atomic: an interrupted run never leaves a truncated part that looks current
    return len(out)


def run(cfg: dict, split: str) -> None:
    """Features stage: compute features for every final candidate pair, part-parallel."""
    t_start = time.perf_counter()
    out_dir = features_dir(cfg, split)
    out_dir.mkdir(parents=True, exist_ok=True)
    pruned = bool(cfg.get("prune", {}).get("enabled"))
    k = None if pruned else cfg["blocking"]["k"]
    parts = sorted(final_candidates_dir(cfg, split).glob("part-*.parquet"))
    if not parts:
        raise SystemExit(f"no candidate parts for {split} in {final_candidates_dir(cfg, split)}")
    version_file = out_dir / "version.txt"
    version = FEATURES_VERSION + (f"+b4:{cfg['blocking']['b4']}" if cfg["blocking"].get("b4") else "")
    if not version_file.exists() or version_file.read_text().strip() != version:
        for old in out_dir.glob("part-*.parquet"):
            old.unlink()  # computed with other feature definitions
        version_file.write_text(version)
    for stale in set(p.name for p in out_dir.glob("part-*.parquet")) - set(p.name for p in parts):
        (out_dir / stale).unlink()  # candidate part no longer exists (re-blocked)
    for partial in out_dir.glob("*.partial"):
        partial.unlink()  # left by an interrupted run
    todo = [(p, out_dir / p.name) for p in parts
            if not ((out_dir / p.name).exists() and (out_dir / p.name).stat().st_mtime > p.stat().st_mtime)]
    if not todo:
        log(f"features {split}: all {len(parts)} parts up to date; skipped")
        return
    s1 = load_normalized(cfg, split, "s1", columns=FEATURE_COLUMNS)
    tg = load_normalized(cfg, split, "targets", columns=FEATURE_COLUMNS)
    competition = target_competition((c if k is None else c[in_candidate_set(c, k)]
                                       for c in map(pd.read_parquet, parts)), len(tg))
    ctx = FeatureContext(s1, tg, competition, load_embeddings(cfg, split))
    ctx.k = k
    ctx.label_codes = None
    if split == "train":
        ts, tt = truth_pairs(cfg, s1["key"].to_numpy(), ctx.t_keys)
        ctx.label_codes = np.sort(ts.astype(np.int64) * len(tg) + tt)
        ctx.roles = s1_roles(cfg, s1["key"].to_numpy())
    del s1, tg
    log(f"features {split}: record tables ready in {time.perf_counter() - t_start:.0f}s"
        f"{' (with embedding features)' if ctx.emb is not None else ''}; {len(todo)} parts to compute")
    procs = n_procs(cfg)
    RF_WORKERS["n"] = max(1, n_threads(cfg) // procs)
    n_pairs = sum(fork_map(_feature_part, ctx, todo, procs))
    log(f"features {split} done: {n_pairs:,} new pairs in {time.perf_counter() - t_start:.0f}s")
