"""Pipeline entry point. Run from code/business_entity_resolution/:

    python -m src.run_pipeline --stage eda
    python -m src.run_pipeline --stage block --step lexical --split train

Stages (in run order; see README.md, and run_v4.sh which overlaps GPU and CPU work):
  eda                                   data checks, raw parquet caches
  normalize                             normalized tables of both splits
  block    --step lexical --split S     TF-IDF blockers (CPU)
  embed    --step train                 fine-tune the bi-encoder (GPU, role-3 S1)
  embed    --step index --split S       encode, B4 search, reverse search (GPU)
  block    --split S                    union of the pool (lexical + B4)
  prune    --step fit                   candidate pruner (stage 0) on role-3 pool pairs
  prune    --step select --split S      keep the best `prune.k` per S1 = the candidate set
  features --split S                    119 pair features + prune_p on the candidate set
  train                                 stage-1 LightGBM, scores of every train and test pair
  rerank   --step train                 mmBERT cross-encoder (GPU)
  rerank   --step score --split all     reranker probabilities of the selected pairs (GPU)
  stack                                 stage-2 graph model
  tune                                  decision rule on validation
  predict                               outputs + validator
  pseudo                                round 2: pseudo-label round, reranker (b) (tag _pl, own outputs)
  raw      [--step data|time|train|score|finish] [--split S]   (no --step = data, train, score, finish)
                                        round 3: raw-text reranker (c) and the stage-2 variants
                                        (own outputs); no --step = every step
  raw      --step handoff|train2|score2 member (d) of the final ensemble (then --step finish
                                        builds the final variant rawpl2)
  all                                   round 1: every stage above except eda, pseudo and raw, in order
                                        (no overlap; normalize loads the raw data itself)

Each stage caches its outputs under the work dir (see src/config.yaml), so re-runs skip
finished work.
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Set before numpy loads (here and in every worker process, which inherits it): OpenBLAS
# otherwise reserves per-thread buffers for all cores in each process. Nothing in this
# pipeline runs dense BLAS work.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from .utils import load_config, log, peak_rss_gb, set_seed  # noqa: E402

STAGES = ["eda", "normalize", "block", "embed", "prune", "features", "train", "stack", "rerank", "tune",
          "predict", "pseudo", "raw", "stress", "all"]
RAW_STEPS = ("data", "time", "train", "score", "finish", "handoff", "train2", "score2")


def pseudo_config(cfg: dict) -> dict:
    """Round 2, the pseudo-label round: its own reranker scores, stage 2, rule and outputs
    (tag "_pl", output_dir/pseudo_label/); everything upstream is shared with the main run."""
    out = dict(cfg)
    out["stack"] = {**cfg["stack"], "rerank_tag": "_pl"}
    out["paths"] = {**cfg["paths"], "output_dir": cfg["paths"]["output_dir"] / "pseudo_label"}
    return out


PL_RAW = [["rr", "_pl"], ["rw", "_raw"]]  # the pseudo-label round's reranker + the raw one
RAW_VARIANTS = {  # stage-2 / decision variants of the raw round: tag -> reranker blocks (prefix, score
    # tag; None = the raw reranker alone), "extra" = another pipeline's per-pair evidence
    # (raw.extra_pairs, stack.attach_extra_pairs; not part of this repository, so those variants are
    # skipped), "assign" = exclusivity on stage-2 p (decide.assignment). The submission is "rawpl2".
    "raw": {"blocks": None},
    "rawpl": {"blocks": PL_RAW},
    "rawboth": {"blocks": [["rr", ""], ["rw", "_raw"]]},   # the main reranker + the raw one
    "plx": {"blocks": [["rr", "_pl"]], "extra": True},       # no raw reranker needed
    "rawplx": {"blocks": PL_RAW, "extra": True},
    "pla": {"blocks": [["rr", "_pl"]], "assign": True},      # no raw reranker needed
    "rawpla": {"blocks": PL_RAW, "assign": True},
    "rawplxa": {"blocks": PL_RAW, "extra": True, "assign": True},
    # with the second raw reranker (train2 / score2 on another machine, scores_<split>_raw2)
    "rawpl2": {"blocks": PL_RAW + [["rx", "_raw2"]]},
    "rawpl2xa": {"blocks": PL_RAW + [["rx", "_raw2"]], "extra": True, "assign": True},
}


def raw_config(cfg: dict, variant: str = "raw") -> dict:
    """Round 3, the raw-text round: a second mmBERT reranker that reads the records as
    shipped (`rerank.text: raw`; models/reranker_raw/, scores_<split>_raw), trained on the
    pairs of `raw.train_s1` role-1 S1 plus, with `raw.pseudo.enabled`, pseudo-labeled test
    pairs from a finished run's decisions (`raw.pseudo.source_tag`, a second self-training
    round). `raw.rerank` overrides reranker settings for this round. Stage 2, rule and outputs under tag "_<variant>" (output_dir/<variant>/),
    see RAW_VARIANTS. Everything upstream is shared with the main run."""
    raw = cfg.get("raw", {})
    out = dict(cfg)
    out["rerank"] = {**cfg["rerank"], "text": "raw", "train_s1": raw.get("train_s1", cfg["rerank"]["train_s1"]),
                     **raw.get("rerank", {})}
    if raw.get("pseudo", {}).get("enabled"):
        out["rerank"]["pseudo_mix"] = {**cfg["pseudo"], **raw["pseudo"]}
    if raw.get("train_frac") is not None:
        out["rerank"]["train_frac"] = float(raw["train_frac"])
    if os.environ.get("BER_RAW_TRAIN_FRAC"):  # run_v4.sh raw: time fallback after the speed check
        out["rerank"]["train_frac"] = float(os.environ["BER_RAW_TRAIN_FRAC"]) * float(raw.get("train_frac", 1.0))
    spec = RAW_VARIANTS[variant]
    out["stack"] = {**cfg["stack"], "rerank_tag": f"_{variant}"}
    if spec["blocks"]:
        out["stack"]["rerank_blocks"] = spec["blocks"]
    if spec.get("extra"):
        out["stack"]["extra_pairs"] = raw["extra_pairs"]
    if spec.get("assign"):
        out["stack"]["full_train_p"] = True
        out["decide"] = {**cfg["decide"], "assignment": True,
                         "assignment_margin": raw.get("assignment_margin", cfg["decide"].get("assignment_margin", 0.0))}
    out["paths"] = {**cfg["paths"], "output_dir": cfg["paths"]["output_dir"] / variant}
    return out


def raw_variant_ready(cfg: dict, variant: str) -> bool:
    """Whether every input of a raw-round stage-2 variant exists (reranker score files, extra
    pair files), so the finish step can build the variants whose inputs are there already."""
    from . import rerank, stack

    c = raw_config(cfg, variant)
    ok = all((rerank.rr_dir(cfg) / f"scores_{sp}{t}.parquet").exists() for _, t in stack.rerank_blocks(c)
             for sp in ("train", "test"))
    for sp in ("train", "test") if RAW_VARIANTS[variant].get("extra") else ():
        p = Path(c["stack"]["extra_pairs"][sp])
        ok &= (p if p.is_absolute() else cfg["paths"]["artifacts_dir"] / p).exists()
    return ok


def main(argv=None) -> int:
    """Parse the CLI, run the requested stage and report its runtime and peak RAM."""
    # Records contain Indian scripts and accented Latin; never crash on a cp1252 console.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--split", default="train", choices=["train", "test", "all"])
    parser.add_argument("--step", default=None,
                        help="block: lexical | union; embed: train | index | baseline | eval | choose | time; "
                             "prune: fit | select; rerank: data | time | train | score | pseudo | score_pl; "
                             "train: fit | test; raw: data | time | train | score | finish | handoff | train2 | score2")
    args = parser.parse_args(argv)

    cfg = load_config()
    set_seed(cfg["seed"])
    log(f"stage={args.stage} step={args.step} split={args.split} work_dir={cfg['paths']['work_dir']}")
    t0 = time.perf_counter()
    splits = ("train", "test") if args.split == "all" else (args.split,)
    if args.stage == "eda":
        from . import eda
        eda.run(cfg)
    elif args.stage == "normalize":
        from .normalize.normalize import load_normalized
        for split in ("train", "test"):
            for side in ("s1", "targets"):
                load_normalized(cfg, split, side, columns=["key"])
    elif args.stage == "block":
        from . import blocking
        for split in splits:
            (blocking.run_lexical if args.step == "lexical" else blocking.run)(cfg, split)
    elif args.stage == "features":
        from . import features
        for split in splits:
            features.run(cfg, split)
    elif args.stage == "embed":
        from . import embed_finetune
        for split in splits:
            embed_finetune.run(cfg, args.step, split)
    elif args.stage == "prune":
        from . import prune
        for split in (splits if args.step != "fit" else ("train",)):
            prune.run(cfg, args.step or "select", split)
    elif args.stage == "stress":
        from . import stress
        stress.run(cfg)
    elif args.stage == "rerank":
        from . import rerank
        rerank.run(cfg, args.step, args.split)
    elif args.stage == "train":  # stage 1: fit and score every train pair (step fit), every test pair (step test)
        from . import train
        if args.step in (None, "fit"):
            train.run(cfg)
        if args.step in (None, "test"):
            train.score_test(cfg)
    elif args.stage == "stack":  # stage 2 on the saved stage-1 predictions
        from . import stack
        stack.run(cfg)
    elif args.stage == "tune":
        from . import decide
        decide.run_tune(cfg)
    elif args.stage == "predict":
        from . import decide
        decide.run_predict(cfg)
    elif args.stage == "pseudo":  # round 2: needs a finished round 1
        from . import decide, rerank, stack
        pcfg = pseudo_config(cfg)
        rerank.pseudo_finetune(pcfg)
        for split in ("train", "test"):
            rerank.score_split(pcfg, split, "_pl")
        stack.run(pcfg)
        decide.run_tune(pcfg)
        decide.run_predict(pcfg)
    elif args.stage == "raw":  # round 3; needs finished rounds 1 and 2 (see raw_config)
        from . import decide, rerank, stack
        rcfg = raw_config(cfg)
        step = args.step or "all"
        if step not in RAW_STEPS + ("all",):
            raise SystemExit(f"unknown raw step '{step}' (one of {', '.join(RAW_STEPS)})")
        if step in ("data", "all"):
            rerank.training_pairs(rcfg, "_raw")
        if step == "time":
            rerank.finetune(rcfg, max_steps=200, tag="_raw")
        if step in ("train", "all"):
            rerank.finetune(rcfg, tag="_raw")
        if step in ("score", "all"):
            for split in (("train", "test") if step == "all" else splits):
                rerank.score_split(rcfg, split, "_raw")
        if step == "handoff":  # member (d): pair lists + training pairs (for this or a second machine)
            rerank.write_handoff(rcfg)
        if step == "train2":  # member (d), the second raw reranker (two-stage recipe)
            rerank.train_two_stage(rcfg, "_raw2")
        if step == "score2":
            scfg = dict(rcfg)
            scfg["rerank"] = {**rcfg["rerank"], **cfg["raw"]["second"].get("rerank", {})}
            for split in splits:
                rerank.score_split(scfg, split, "_raw2")
        if step in ("finish", "all"):  # every stage-2 variant whose inputs exist; pick by validation
            only = os.environ.get("BER_RAW_VARIANTS", "").split(",") if os.environ.get("BER_RAW_VARIANTS") else None
            for variant in RAW_VARIANTS:
                if only and variant not in only:
                    continue
                if not raw_variant_ready(cfg, variant):
                    log(f"raw variant {variant}: inputs missing; skipped")
                    continue
                c = raw_config(cfg, variant)
                log(f"=== raw variant {variant}: stage 2, tune, predict -> {c['paths']['output_dir']} ===")
                stack.run(c)
                decide.run_tune(c)
                decide.run_predict(c)
    elif args.stage == "all":
        from . import blocking, decide, embed_finetune, features, prune, rerank, stack, train
        from .normalize.normalize import load_normalized
        for split in ("train", "test"):
            for side in ("s1", "targets"):
                load_normalized(cfg, split, side, columns=["key"])
        for split in ("train", "test"):
            blocking.run_lexical(cfg, split)
        if cfg["blocking"].get("b4"):
            if cfg["blocking"]["b4"] == "ft":
                embed_finetune.finetune(cfg)
            for split in ("train", "test"):
                embed_finetune.run(cfg, "index", split)
        for split in ("train", "test"):
            blocking.run(cfg, split)
        if cfg.get("prune", {}).get("enabled"):
            prune.fit(cfg)
            for split in ("train", "test"):
                prune.select(cfg, split)
        for split in ("train", "test"):
            features.run(cfg, split)
        train.run(cfg)
        train.score_test(cfg)
        if cfg["stack"].get("enabled"):
            if cfg["stack"].get("rerank"):
                rerank.finetune(cfg)
                for split in ("train", "test"):
                    rerank.score_split(cfg, split)
            stack.run(cfg)
        decide.run_tune(cfg)
        decide.run_predict(cfg)
    else:
        raise SystemExit(f"stage '{args.stage}' is not implemented yet")
    log(f"stage {args.stage} finished in {time.perf_counter() - t0:.1f}s, "
        f"peak RSS {peak_rss_gb():.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
