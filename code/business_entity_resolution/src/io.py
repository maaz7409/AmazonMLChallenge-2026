"""Loading raw challenge files, caching them as parquet, and parsing entity IDs.

Every raw TSV is read with exactly the call the challenge prescribes: tab separator,
all columns as strings, empty fields kept as "" (never NaN) and no quote processing,
so values such as `"Joe's" Pizza` survive verbatim.

Entity IDs such as "S2-166376419" are also mapped to int64 keys
(source * 10**10 + number). Keys are unique across sources and much cheaper to join
on than strings, which matters with ~10M target records per split.
"""

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
GT_COLUMNS = ["source1_entity_id", "matched_entity_ids"]
KEY_BASE = 10**10
# No leading zeros allowed, so every int key maps back to exactly one ID string.
_ID_REGEX = r"^S[123]-(0|[1-9][0-9]{0,9})$"


def read_tsv(path) -> pd.DataFrame:
    """Read a challenge TSV with the prescribed, lossless settings."""
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                       quoting=csv.QUOTE_NONE)


def count_data_lines(path) -> int:
    """Count the data rows of a raw file (all lines minus the header) on raw bytes."""
    n_newlines, last = 0, b"\n"
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            n_newlines += chunk.count(b"\n")
            last = chunk[-1:]
    return n_newlines + (last != b"\n") - 1  # a final line may lack its newline


def source_path(cfg: dict, split: str, source: int) -> Path:
    """Path of a raw source file, e.g. dataset/train/train_source2.tsv."""
    return cfg["paths"]["data_dir"] / split / f"{split}_source{source}.tsv"


def ground_truth_path(cfg: dict) -> Path:
    """Path of the training ground-truth file."""
    return cfg["paths"]["data_dir"] / "train" / "train_ground_truth.tsv"


def raw_cache_path(cfg: dict, name: str) -> Path:
    """Parquet cache of a raw file, named by its stem (e.g. 'train_source2')."""
    return cfg["paths"]["artifacts_dir"] / "raw" / f"{name}.parquet"


def write_raw_cache(cfg: dict, name: str, df: pd.DataFrame) -> None:
    """Write a raw file's frame to its parquet cache so later stages skip TSV parsing."""
    path = raw_cache_path(cfg, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, compression="zstd")


def _load_cached(cfg: dict, name: str, tsv: Path, columns) -> pd.DataFrame:
    cache = raw_cache_path(cfg, name)
    if cache.exists():
        return pd.read_parquet(cache, columns=columns)
    df = read_tsv(tsv)
    write_raw_cache(cfg, name, df)
    return df if columns is None else df[columns]


def load_source(cfg: dict, split: str, source: int, columns=None) -> pd.DataFrame:
    """Load one source file (S1/S2/S3 of a split), via its parquet cache when present."""
    return _load_cached(cfg, f"{split}_source{source}", source_path(cfg, split, source), columns)


def load_ground_truth(cfg: dict) -> pd.DataFrame:
    """Load the training ground truth, via its parquet cache when present."""
    return _load_cached(cfg, "train_ground_truth", ground_truth_path(cfg), None)


def as_arrow(values) -> pa.Array:
    """Return a contiguous pyarrow large_string array for a Series, array or Arrow input.

    Arrow-backed pandas string columns convert without copying the strings; one string
    type throughout lets arrays from different sources be concatenated.
    """
    if not isinstance(values, (pa.Array, pa.ChunkedArray)):
        values = pa.array(values if isinstance(values, pd.Series) else
                          np.asarray(values, dtype=object))
    if isinstance(values, pa.ChunkedArray):
        values = values.combine_chunks()
    return values.cast(pa.large_string())


def id_keys(ids) -> np.ndarray:
    """Map entity IDs like 'S2-166376419' to int64 keys (source * 10**10 + number).

    Raises ValueError if any ID is not 'S<1|2|3>-<digits without leading zeros>'.
    """
    arr = as_arrow(ids)
    ok = pc.match_substring_regex(arr, _ID_REGEX)
    if not pc.all(ok).as_py():
        bad = arr.filter(pc.invert(ok)).slice(0, 5).to_pylist()
        raise ValueError(f"unexpected entity_id format, e.g. {bad}")
    src = pc.cast(pc.utf8_slice_codeunits(arr, 1, 2), pa.int64()).to_numpy()
    num = pc.cast(pc.utf8_slice_codeunits(arr, 3), pa.int64()).to_numpy()
    return src * KEY_BASE + num


def key_to_id(key: int) -> str:
    """Inverse of id_keys for one key, e.g. 20166376419 -> 'S2-166376419'."""
    return f"S{key // KEY_BASE}-{key % KEY_BASE}"


def positions(haystack: np.ndarray, needles: np.ndarray) -> np.ndarray:
    """Index of each needle in `haystack` (unique keys), or -1 when absent."""
    order = np.argsort(haystack, kind="stable")
    ordered = haystack[order]
    pos = np.minimum(np.searchsorted(ordered, needles), len(ordered) - 1)
    return np.where(ordered[pos] == needles, order[pos], -1)


def hash_uniform(keys: np.ndarray, salt: int) -> np.ndarray:
    """Deterministic pseudo-uniform [0, 1) value per int64 key (splitmix64 of key ^ salt)."""
    with np.errstate(over="ignore"):
        z = (keys.astype(np.uint64) ^ np.uint64(salt)) + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return (z >> np.uint64(11)).astype(np.float64) / 2.0**53


def s1_roles(cfg: dict, s1_keys: np.ndarray) -> np.ndarray:
    """Role of each train S1 record: 2 = validation, 1 = stage-1 training sample,
    3 = embedder (and candidate pruner) training sample, 0 = the rest (stage 2 samples it).

    Validation is `split.val_frac` of S1 IDs by seeded hash. Stage 1 fits on a seeded sample
    of `split.train_s1` records from the remaining ones; a disjoint sample of `split.embed_s1`
    records trains the bi-encoder and the pruner, so every embedding feature stage 1, stage 2,
    validation and test see is out-of-sample for the embedder (an embedder trained on stage 1's
    own S1 records would give stage 1 optimistic cosines).
    """
    scfg = cfg["split"]
    u = hash_uniform(s1_keys, cfg["seed"])
    roles = np.zeros(len(s1_keys), dtype=np.int8)
    roles[u < scfg["val_frac"]] = 2
    pool = u >= scfg["val_frac"]
    n_pool = max(1, int(pool.sum()))
    rate1 = min(1.0, scfg["train_s1"] / n_pool)
    rate3 = min(1.0 - rate1, scfg.get("embed_s1", 0) / n_pool)
    u1 = hash_uniform(s1_keys, cfg["seed"] + 1)
    roles[pool & (u1 < rate1)] = 1
    roles[pool & (u1 >= rate1) & (u1 < rate1 + rate3)] = 3
    return roles


def truth_pairs(cfg: dict, s1_keys: np.ndarray, t_keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ground-truth pairs as (S1 row, target row) positions in the normalized train tables."""
    gt = load_ground_truth(cfg)
    rows, keys, _ = split_id_lists(gt["matched_entity_ids"])
    s1_pos = positions(s1_keys, id_keys(gt["source1_entity_id"]))
    return s1_pos[rows].astype(np.int32), positions(t_keys, keys).astype(np.int32)


def write_id_lists(path: Path, header: list[str], s1_ids: list[str], lists: list[list[str]]) -> None:
    """Write a submission-style TSV: one row per S1 ID, comma-joined IDs (possibly empty).

    Written by hand rather than through pandas so nothing can be quoted; every field is
    checked to contain no tab, comma-free S1 IDs and no newline.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(header) + "\n")
        for s1, ids in zip(s1_ids, lists):
            joined = ",".join(ids)
            if "\t" in s1 or "\t" in joined or "\n" in joined or "," in s1:
                raise ValueError(f"illegal character in output row for {s1}")
            f.write(f"{s1}\t{joined}\n")


def split_id_lists(values) -> tuple[np.ndarray, np.ndarray, int]:
    """Split a column of comma-separated ID lists into flat (row, key) arrays.

    Returns (row position of each ID, its int64 key, number of stray empty tokens
    inside non-empty lists such as "a,,b"). Empty lists contribute no entries.
    """
    arr = as_arrow(values)
    lists = pc.split_pattern(arr, ",")
    rows = pc.list_parent_indices(lists).to_numpy()
    flat = pc.list_flatten(lists)
    keep = pc.not_equal(flat, "")
    n_empty_tokens = len(flat) - pc.sum(keep).as_py()
    n_empty_values = pc.sum(pc.equal(arr, "")).as_py() or 0
    keys = id_keys(flat.filter(keep))
    return rows[keep.to_numpy(zero_copy_only=False)], keys, n_empty_tokens - n_empty_values
