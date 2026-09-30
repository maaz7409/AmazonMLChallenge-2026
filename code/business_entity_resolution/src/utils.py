"""Shared runtime helpers: configuration, seeding, progress logging and memory tracking."""

import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
_START = time.perf_counter()


def load_config(path: Path | None = None) -> dict:
    """Load config.yaml and resolve every entry under `paths` to an absolute Path.

    Relative paths are taken relative to the repository root. The BER_CONFIG environment
    variable, when set, names another config file (a run-specific copy); BER_WORK_DIR,
    BER_DATA_DIR and BER_OUTPUT_DIR, when set, override `paths.work_dir`, `paths.data_dir`
    and `paths.output_dir`; `artifacts_dir` and `models_dir` are derived from the work dir.
    """
    path = Path(path or os.environ.get("BER_CONFIG") or CONFIG_PATH)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    paths = cfg["paths"]
    for env, key in (("BER_WORK_DIR", "work_dir"), ("BER_DATA_DIR", "data_dir"),
                     ("BER_OUTPUT_DIR", "output_dir")):
        if os.environ.get(env):
            paths[key] = os.environ[env]
    for key, value in list(paths.items()):
        p = Path(value)
        paths[key] = (p if p.is_absolute() else REPO_ROOT / p).resolve()
    paths["artifacts_dir"] = paths["work_dir"] / "artifacts"
    paths["models_dir"] = paths["work_dir"] / "models"
    return cfg


def n_workers(cfg: dict) -> int:
    """Worker processes for CPU-parallel stages (config `workers`, default: all cores)."""
    return int(cfg.get("workers") or os.cpu_count() or 2)


def n_threads(cfg: dict) -> int:
    """Threads for multi-threaded native code: LightGBM, sparse top-K, rapidfuzz.

    Always passed explicitly: rented GPU boxes often export OMP_NUM_THREADS=1, which
    otherwise makes LightGBM training and prediction single-threaded.
    """
    return int(cfg.get("threads") or os.cpu_count() or 2)


def device() -> str:
    """'cuda' when a GPU is visible, else 'cpu' (smoke tests on CPU boxes)."""
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int) -> None:
    """Seed Python's and NumPy's global RNGs (model-specific seeds are set where used)."""
    random.seed(seed)
    np.random.seed(seed)


def rss_gb() -> float:
    """Current resident memory of this process, in GB."""
    return psutil.Process().memory_info().rss / 2**30


def peak_rss_gb() -> float:
    """Peak resident memory of this process since it started, in GB.

    Uses the peak working set on Windows and ru_maxrss on Linux/macOS, so it also
    catches short spikes inside C code that a polling thread would miss.
    """
    info = psutil.Process().memory_info()
    if hasattr(info, "peak_wset"):  # Windows
        return info.peak_wset / 2**30
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2**30 if sys.platform == "darwin" else peak / 2**20  # bytes on macOS, KB on Linux


def log(msg: str) -> None:
    """Print a progress line prefixed with elapsed seconds, current and peak memory."""
    print(f"[{time.perf_counter() - _START:7.1f}s | rss {rss_gb():5.1f} GB | "
          f"peak {peak_rss_gb():5.1f} GB] {msg}", flush=True)


_FORK_CTX = {}


def _fork_call(args):
    fn, item = args
    return fn(_FORK_CTX["ctx"], item)


def fork_map(fn, ctx, items: list, n_proc: int) -> list:
    """[fn(ctx, item) for item in items] in `n_proc` forked worker processes (Linux).

    The context (large read-only tables, token indexes, memory-mapped embeddings) is built
    once in the parent and inherited copy-on-write, so workers start instantly and share its
    memory. `fn` must be a module-level function and should write its big outputs to disk
    and return something small. Nothing that starts OpenMP threads (LightGBM) may run in the
    parent before this call: libgomp is not fork-safe once its pool exists. Falls back to a
    plain loop for one process or on platforms without fork.
    """
    import multiprocessing as mp

    if n_proc <= 1 or len(items) <= 1 or "fork" not in mp.get_all_start_methods():
        return [fn(ctx, it) for it in items]
    _FORK_CTX["ctx"] = ctx
    try:
        with mp.get_context("fork").Pool(min(n_proc, len(items))) as pool:
            return list(pool.imap_unordered(_fork_call, [(fn, it) for it in items]))
    finally:
        _FORK_CTX.clear()


def n_procs(cfg: dict) -> int:
    """Forked processes for part-parallel CPU stages (config `part_procs`)."""
    return int(cfg.get("part_procs") or max(1, min(8, (os.cpu_count() or 2) // 4)))
