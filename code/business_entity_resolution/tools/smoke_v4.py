"""End-to-end smoke test of the pipeline on a tiny SYNTHETIC dataset (CPU is enough).

    python tools/smoke_v4.py [--work /tmp/ber_smoke_v4] [--keep] [--pseudo]

Writes a synthetic dataset (tools/make_synthetic.py), a config derived from src/config.yaml
with tiny public test models (a 4M-parameter sentence-transformer for B4, a random tiny BERT
as the reranker) and small samples, then runs `bash run_v4.sh` (the real runner, with its
GPU/CPU chains) twice: the second run must skip the expensive steps. With --pseudo it also
runs the optional pseudo-label round. Every output must pass the official validator. The
scores mean nothing; this only proves that every stage runs, resumes and writes valid files.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]


def smoke_config(work: Path) -> Path:
    """Write the smoke-test config next to the synthetic data and return its path."""
    cfg = yaml.safe_load((ROOT / "src" / "config.yaml").read_text(encoding="utf-8"))
    cfg["split"].update(train_s1=1200, embed_s1=800)
    cfg["blocking"].update(s1_per_part=500, chunk=500)
    cfg["embed"].update(model="sentence-transformers-testing/stsb-bert-tiny-safetensors", dim=128,
                        max_seq_length=32, encode_batch=128, batch_size=32, mini_batch_size=16, eval_queries=100)
    cfg["rerank"].update(model="hf-internal-testing/tiny-random-bert", train_s1=400, batch_size=16,
                         score_batch=64, max_length=48)
    cfg["prune"].update(max_rounds=100)
    cfg["stack"].update(train_s1=900, max_rounds=200)
    cfg["lgbm"]["max_rounds"] = 200
    cfg["pseudo"].update(unseen_s1=400, seen_s1=200, hi=0.5, lo=0.5, lo_single=0.5)
    cfg.update(workers=2, threads=2, part_procs=2)
    path = work / "smoke_config.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def run(cmd: list[str], env: dict) -> None:
    """Run a command in the repository root, fail loudly on error."""
    print("$", " ".join(cmd), flush=True)
    t0 = time.perf_counter()
    res = subprocess.run(cmd, cwd=REPO, env=env)
    if res.returncode != 0:
        raise SystemExit(f"FAILED: {' '.join(cmd)}")
    print(f"  ({time.perf_counter() - t0:.0f}s)", flush=True)


def main() -> None:
    """Build the synthetic universe, run the pipeline twice (resume check), check outputs."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/tmp/ber_smoke_v4")
    ap.add_argument("--keep", action="store_true", help="reuse an existing work dir (resume check only)")
    ap.add_argument("--pseudo", action="store_true", help="also run the optional pseudo-label round")
    args = ap.parse_args()
    work = Path(args.work)
    if work.exists() and not args.keep:
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    if not (work / "dataset" / "train").exists():
        subprocess.run([sys.executable, str(ROOT / "tools" / "make_synthetic.py"), "--out", str(work / "dataset"),
                        "--n-train", "4000", "--n-test", "3000"], check=True)
    env = {**os.environ, "BER_CONFIG": str(smoke_config(work)), "BER_DATA_DIR": str(work / "dataset"),
           "BER_WORK_DIR": str(work / "work"), "BER_OUTPUT_DIR": str(work / "output"),
           "BER_LOG_DIR": str(work / "logs"), "PYTHON": sys.executable,
           "OMP_NUM_THREADS": "1"}  # as on rented GPU boxes: LightGBM must still use its threads
    run(["bash", "run_v4.sh"], env)
    log = (work / "logs" / "predict.log").read_text(encoding="utf-8")
    assert "validator exit code 0" in log and "streaming check: PASS" in log, "validator did not pass"
    first = (work / "output" / "matching_results.tsv").read_bytes()
    run(["bash", "run_v4.sh"], env)  # resume: every expensive step must be skipped
    logs = {p.stem: p.read_text(encoding="utf-8") for p in (work / "logs").glob("*.log")}
    for name, marker in (("embed_train", "already fine-tuned"), ("index_train", "already indexed"),
                         ("union_train", "already built"), ("prune_fit", "already fitted"),
                         ("select_test", "up to date"), ("features_train", "up to date"),
                         ("stage1_fit", "already fitted and scored"), ("rerank_train", "already trained"),
                         ("rerank_score_test", "up to date")):
        assert marker in logs[name], f"second run did not skip {name}"
    assert (work / "output" / "matching_results.tsv").read_bytes() == first, "second run changed the output"
    if args.pseudo:
        run(["bash", "run_v4.sh", "pseudo"], env)
        plog = (work / "logs" / "pseudo.log").read_text(encoding="utf-8")
        assert "validator exit code 0" in plog, "pseudo-label outputs did not validate"
    print(f"smoke test passed (logs: {work / 'logs'})")


if __name__ == "__main__":
    main()
