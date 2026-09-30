# Business Entity Resolution: Team Ballin (Amazon ML Challenge 2026)

**Team:** Ballin ( [@maaz7409](https://github.com/maaz7409)
[@AnshulPatil2005](https://github.com/AnshulPatil2005)
[@KeshavKumar-0](https://github.com/KeshavKumar-0)
[@Nipun-Shekhar](https://github.com/Nipun-Shekhar) )

**Submission:** 29 September 2026

This is the complete guide to our solution: how to run it (sections 1-8) and how it works
(sections 9-13). The task is described in [`../../PROBLEM_STATEMENT.md`](../../PROBLEM_STATEMENT.md).
A beginner-friendly walkthrough of the same pipeline is in
[`../../PIPELINE_EXPLAINED.md`](../../PIPELINE_EXPLAINED.md).

| | |
| --- | --- |
| Final submission | variant **`rawpl2`**: a graph-aware LightGBM (stage 2) over an ensemble of three mmBERT cross-encoders |
| Result | public leaderboard macro F0.5 **0.988**, rank **287** (of ~25,000 registered teams) |
| Entry point | `python -m src.run_pipeline --stage <stage> [--step <step>] [--split train\|test\|all]`, run from this folder |
| Models | `Snowflake/snowflake-arctic-embed-m-v2.0` (Apache-2.0, ~305M), `jhu-clsp/mmBERT-base` (MIT, 307M), LightGBM (MIT) |
| External data | none: only the challenge files |

**Contents**

- Running it: [1. Quick start](#1-quick-start) · [2. Repository layout](#2-repository-layout) ·
  [3. Hardware](#3-hardware) · [4. Setup](#4-setup) · [5. Quick check without a GPU](#5-quick-check-without-a-gpu-2-3-min) ·
  [6. Reproduce the submission](#6-reproduce-the-submission-rawpl2) · [7. Run time](#7-run-time) ·
  [8. Outputs and checks](#8-outputs-and-checks)
- How it works: [9. Summary](#9-summary) · [10. Methodology](#10-methodology) ·
  [11. Results and error analysis](#11-results-and-error-analysis) ·
  [12. The final ensemble (`rawpl2`)](#12-the-final-ensemble-rawpl2) · [13. Variants and the majority vote](#13-variants-and-the-majority-vote)
- Reference: [14. Configuration and troubleshooting](#14-configuration-and-troubleshooting) ·
  [15. Code map](#15-code-map) · [16. Fair play and licenses](#16-fair-play-and-licenses) ·
  [17. Conclusion](#17-conclusion)

---

## 1. Quick start

```bash
# at the repository root, with the challenge data in dataset/train and dataset/test
uv venv --python 3.12 .venv && uv pip install --python .venv -r code/business_entity_resolution/requirements.txt
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0                     # exactly one GPU (section 3)

bash run_v4.sh            # round 1: main pipeline                -> output/matching_results.tsv, candidate_pairs.tsv
bash run_v4.sh pseudo     # round 2: self-trained reranker (b)    -> output/pseudo_label/
bash run_v4.sh raw        # round 3: raw-text reranker (c)        -> output/rawpl/ and other variants
cd code/business_entity_resolution                # ensemble member (d), then the final stage 2
python -m src.run_pipeline --stage raw --step handoff
python -m src.run_pipeline --stage raw --step train2
python -m src.run_pipeline --stage raw --step score2 --split all
BER_RAW_VARIANTS=rawpl2 python -m src.run_pipeline --stage raw --step finish   # -> output/rawpl2/
cd ../..
bash run_final.sh         # copies rawpl2 to output/matching_results.tsv and validates (all rounds cached)
```

Every step caches its result, so any command can be re-run after a crash and continues where it
stopped. Section 6 explains each command; section 7 lists the run times (~21-24 h on one L4).

---

## 2. Repository layout

```
<repository root>/
├── README.md                     short overview
├── PROBLEM_STATEMENT.md          the challenge description
├── PIPELINE_EXPLAINED.md         beginner-friendly walkthrough and fact sheet
├── run_v4.sh                     runner: round 1 (default), round 2 (`pseudo`), round 3 (`raw`)
├── run_final.sh                  rounds 1-3, then copies the chosen variant to output/ and validates
├── utils/validate_submission.py  official submission validator
├── output/
│   ├── matching_results.tsv      final matches (variant rawpl2)
│   └── candidate_pairs.tsv.xz    candidate set, xz-compressed (~90 MB instead of ~280 MB) so that it
│                                 fits on GitHub; extract it first (section 8)
├── logs/v4/                      logs of our round-1 and round-2 runs (a new run appends here)
├── dataset/                      <- put the challenge data here (train/, test/); not included
└── code/business_entity_resolution/
    ├── README.md                 this guide
    ├── requirements.txt          pinned environment (Python 3.12)
    ├── src/                      pipeline source; src/config.yaml holds every setting
    ├── tools/                    smoke test, synthetic data, majority vote, diagnostics
    ├── experiment_log.md         results of the final pipeline's runs and the design experiments
    └── reports/data_report.md    data checks (row counts, countries, singletons, exclusivity)
```

Paths below are relative to the **repository root** unless a command starts with
`cd code/business_entity_resolution`.

---

## 3. Hardware

**What we used:** one NVIDIA **L4** (24 GB) per machine, 256 vCPU, 128 GB+ RAM, Ubuntu. We ran
ensemble member (d) on a second L4 only to save time; **one GPU is enough**.

**What the code needs:** one CUDA GPU with **>= 24 GB** and **bf16 support** (Ampere or newer:
A10G, L4, L40S, A100, H100, ...). Nothing is tuned to the L4: a bigger single GPU such as an A100
or H100 runs the same code unchanged, and faster. The training batch sizes are fixed in
`src/config.yaml`, so a bigger GPU trains the same models as ours.

| | needed |
| --- | --- |
| GPU | 1x CUDA GPU, >= 24 GB, bf16-capable (Ampere or newer). T4 / V100 are **not** supported (no bf16) |
| NVIDIA driver | >= 580 for the default torch build (CUDA 13.0); older drivers: section 14 |
| CPUs | 28+ vCPU; more is faster (blocking, features and the pruner run on the CPU) |
| RAM | ~128 GB |
| Disk | ~150 GB free |
| Python | **3.12** (numpy 2.5 / pandas 3 need it) |
| Network | first run only: pip packages and the two Hugging Face models (~2 GB) |

> **Multi-GPU machines (e.g. AWS p4d / p5): expose exactly one GPU.**
> `export CUDA_VISIBLE_DEVICES=0` before every command. With several visible GPUs, the Hugging Face
> trainers spread training over all of them, which multiplies the effective batch size and changes
> the models.

---

## 4. Setup

```bash
# at the repository root
# 1. data: the challenge's train/ and test/ folders as shipped
mkdir -p dataset && cp -r /path/to/challenge/dataset/{train,test} dataset/
ls dataset/train dataset/test   # train_source{1,2,3}.tsv, train_ground_truth.tsv / test_source{1,2,3}.tsv

# 2. environment: a Python 3.12 venv at .venv (run_v4.sh and run_final.sh activate it themselves)
uv venv --python 3.12 .venv
uv pip install --python .venv -r code/business_entity_resolution/requirements.txt
#    without uv: python3.12 -m venv .venv && .venv/bin/pip install -r code/business_entity_resolution/requirements.txt
source .venv/bin/activate
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
#    expected: 2.14.0+cu130 True <your GPU>

# 3. one GPU only (section 3)
export CUDA_VISIBLE_DEVICES=0
```

**CPU settings.** `src/config.yaml` is set for our 256-CPU machine: `threads: 64`, `workers: 24`,
`part_procs: 6`. Set `threads` to the machine's CPU count if it has fewer than 64. These three
settings change speed only, never results. Each `part_procs` worker needs ~3-5 GB RAM.

---

## 5. Quick check without a GPU (~2-3 min)

Runs the real runner (`run_v4.sh`) end to end on a small **synthetic** dataset with tiny test
models, twice (the second run must skip every finished step and change nothing), plus round 2,
and validates every output. The scores mean nothing; it only proves that the round 1 and round 2
stages run. Round 3 (`raw`), the ensemble steps, `tools/vote.py` and `run_final.sh` are not covered.

```bash
cd code/business_entity_resolution && python tools/smoke_v4.py --pseudo && cd ../..
```

---

## 6. Reproduce the submission (`rawpl2`)

The pipeline runs in three rounds plus the ensemble step; each round writes a complete, valid
submission of its own:

| round | adds | output | public LB |
| --- | --- | --- | --- |
| 1 | the main pipeline (blocking, pruner, stage 1, reranker (a), stage 2, rule) | `output/matching_results.tsv` | 0.984 |
| 2 | reranker (b): (a) self-trained on confident test decisions | `output/pseudo_label/` | 0.986 |
| 3 | reranker (c): raw text, second self-training round; stage-2 variants | `output/rawpl/` (and others) | 0.988 |
| ensemble | reranker (d): a second raw-text model; stage 2 over (b) + (c) + (d) | `output/rawpl2/` | **0.988** |

Run inside `tmux` or `screen`. Every step caches its output under
`code/business_entity_resolution/artifacts/` and `models/`: after a crash or reboot, run the same
command again and finished steps are skipped.

### 6.1 With the runner scripts (recommended: CPU and GPU steps overlap)

```bash
# at the repository root, venv active, CUDA_VISIBLE_DEVICES=0
bash run_v4.sh            # round 1: main pipeline              -> output/matching_results.tsv, output/candidate_pairs.tsv
bash run_v4.sh pseudo     # round 2: self-trained reranker (b)  -> output/pseudo_label/
bash run_v4.sh raw        # round 3: raw-text reranker (c)      -> output/rawpl/ and other variants

cd code/business_entity_resolution          # ensemble member (d), then the final stage 2
python -m src.run_pipeline --stage raw --step handoff              # pair lists for member (d)
python -m src.run_pipeline --stage raw --step train2               # member (d): two-stage recipe, seed 43
python -m src.run_pipeline --stage raw --step score2 --split all   # member (d) scores, train and test
BER_RAW_VARIANTS=rawpl2 python -m src.run_pipeline --stage raw --step finish   # -> output/rawpl2/
cd ../..

bash run_final.sh         # copies rawpl2 to output/ and validates (all steps cached: quick)
```

- `run_final.sh` runs rounds 1-3 (cached here) and then copies `rawpl2` by default;
  `BER_FINAL_VARIANT=<variant>` picks another (section 13). Run on a fresh machine before the
  ensemble steps, it stops at the copy step because `output/rawpl2/` does not exist yet.
- `bash run_v4.sh raw` logs `raw variant rawpl2: inputs missing; skipped` until the four ensemble
  commands have run; that is expected.
- Progress: `tail -f logs/v4/progress.log` (rounds 1-2), `tail -f logs/v4/raw/progress.log`
  (round 3); every step also has its own log next to them.

### 6.2 Step by step (same stages, one at a time)

The same pipeline without the runner scripts, e.g. to follow or time each stage. From
`code/business_entity_resolution/`:

```bash
# Round 1: main pipeline
python -m src.run_pipeline --stage eda                                  # data checks, parquet caches
python -m src.run_pipeline --stage normalize                            # normalized name/address views
python -m src.run_pipeline --stage block --step lexical --split all     # word + char TF-IDF blockers (CPU)
python -m src.run_pipeline --stage embed --step train                   # fine-tune the arctic-embed blocker (GPU)
python -m src.run_pipeline --stage embed --step index --split all       # encode, top-30 search (GPU)
python -m src.run_pipeline --stage block --split all                    # union of the blockers: the pool
python -m src.run_pipeline --stage prune --step fit                     # candidate pruner
python -m src.run_pipeline --stage prune --step select --split all      # the candidate set
python -m src.run_pipeline --stage features --split all                 # 119 pair features
python -m src.run_pipeline --stage train                                # stage 1 (LightGBM)
python -m src.run_pipeline --stage rerank --step train                  # mmBERT reranker (a) (GPU)
python -m src.run_pipeline --stage rerank --step score --split all      # its scores (GPU)
python -m src.run_pipeline --stage stack                                # stage 2
python -m src.run_pipeline --stage tune                                 # decision rule on validation
python -m src.run_pipeline --stage predict                              # -> ../../output/{matching_results,candidate_pairs}.tsv

# Round 2: self-trained reranker (b)
python -m src.run_pipeline --stage pseudo                               # -> ../../output/pseudo_label/

# Round 3: raw-text reranker (c)
python -m src.run_pipeline --stage raw                                  # -> ../../output/rawpl/ and other variants

# Ensemble member (d), then the final stage 2
python -m src.run_pipeline --stage raw --step handoff
python -m src.run_pipeline --stage raw --step train2
python -m src.run_pipeline --stage raw --step score2 --split all
BER_RAW_VARIANTS=rawpl2 python -m src.run_pipeline --stage raw --step finish   # -> ../../output/rawpl2/

# Final file (candidate_pairs.tsv from round 1 is shared by every round)
cp ../../output/rawpl2/matching_results.tsv ../../output/matching_results.tsv
```

`python -m src.run_pipeline --stage all` runs round 1 in one process, without the CPU/GPU overlap.

### 6.3 Optional: member (d) on a second GPU machine

This is how we saved time; it is not needed on one fast GPU. After round 3's data step
(`raw_data` in `logs/v4/raw/progress.log`), run `handoff` on the main machine. Copy
`artifacts/rerank/handoff_pairs_{train,test}.parquet` and `artifacts/rerank/train_pairs_raw.parquet`
to a second machine with the same code and dataset, and run `train2` and `score2 --split all`
there. Copy `artifacts/rerank/scores_{train,test}_raw2.parquet` back, then run `finish` as above.

---

## 7. Run time

Measured and estimated on our L4 machines (256 vCPU). The GPU steps are the long ones; on a
faster single GPU (A100 / H100) they should run several times faster. We have not timed such a
GPU, and the CPU steps depend on the number of vCPUs, not on the GPU.

| step | on one L4 | |
| --- | --- | --- |
| round 1 (`bash run_v4.sh`) | ~7.5-8 h | measured (`logs/v4/progress.log`). Longest steps: embedder fine-tune 76 min, encoding + search 89 + 77 min, reranker training 51 min, reranker scoring 43 + 107 min, lexical blocking 31 + 17 min (CPU), stage-1 fit 41 min (CPU) |
| round 2 (`bash run_v4.sh pseudo`) | ~2 h | measured |
| round 3 (`bash run_v4.sh raw`) | ~5-6 h | estimate: training ~2.5-3 h, scoring ~2-2.5 h, variants ~25 min |
| member (d) (`handoff`, `train2`, `score2`) | ~6-7 h | estimate: twice round 3's share of training pairs, two training stages, scoring |
| `finish` + `run_final.sh` | ~20 min | |
| **total** | **~21-24 h on one L4** | about 16-17 h with member (d) on a second L4 |

Shorter checks on the real data: round 1 alone gives a complete submission (LB 0.984) in
~7.5-8 h; the steps of section 6.2 up to `--stage prune --step select` give the candidate set alone
in ~4.5-5 h; rounds 1-3 plus
`BER_FINAL_VARIANT=rawpl bash run_final.sh` give `rawpl` (LB 0.988) in ~15-16 h.

---

## 8. Outputs and checks

| file | content |
| --- | --- |
| `output/matching_results.tsv` | final matches, one row per test Source-1 record (copy of `output/rawpl2/`) |
| `output/candidate_pairs.tsv` | the pruned candidate set every model scores (~19.9M pairs, ~11.5 per record); the same for every round and variant. Shipped in this repository as `output/candidate_pairs.tsv.xz` (extract it, see below); a fresh run of round 1 writes the plain `.tsv` itself |
| `output/<variant>/matching_results.tsv` | each round and variant: `pseudo_label`, `raw`, `rawpl`, `rawboth`, `pla`, `rawpla`, `rawpl2` |
| `logs/v4/tune.log`, `logs/v4/pseudo.log`, `logs/v4/raw/raw_finish.log` | `tuned decision ...`: validation macro F0.5 (US / India) of rounds 1, 2 and 3 |
| `code/business_entity_resolution/artifacts/candidates/{train,test}_pruned/prune_report.json` | candidate counts; on train also the validation recall of the candidate set |
| `code/business_entity_resolution/artifacts/submissions/<timestamp>/` | archive of every `predict` run |

**Extracting the candidate file.** `output/candidate_pairs.tsv` is 265 MiB, so the repository ships
it xz-compressed (89.5 MiB) as `output/candidate_pairs.tsv.xz`, small enough to push to GitHub.
Extract it before validating or submitting (at the repository root; `-k` keeps the archive):

```bash
xz -dk output/candidate_pairs.tsv.xz     # -> output/candidate_pairs.tsv
```

Every `predict` runs the official validator on what it writes (`validator exit code 0` and
`candidate_pairs.tsv streaming check: PASS` in its log). To check by hand:

```bash
python3 utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids
```

**Reproducibility.** Seeds are fixed (42; member (d) 43). Re-running a command reuses the cached
steps, so outputs stay byte-identical. A run from scratch can differ slightly from ours, because
GPU training of the transformer models is not bit-for-bit deterministic (and differs between GPU
models); expect small differences in the candidate set, the matches and the score.

**The candidate file.** The run that produced `rawpl2` pruned 84,172,807 pool pairs to
**19,944,225** test candidates (11.51 per record; `logs/v4/select_test.log`); every model scored
exactly that set. Its `candidate_pairs.tsv` was lost with that machine, so the file that goes with
`output/matching_results.tsv` was regenerated from the same code (round 1's steps up to the
pruner, section 6.2): **19,835,499** pairs (11.45 per record). Because of the non-determinism above, 631 of the
5,862,403 final matches (0.011 %) lie outside the regenerated set.

---

## 9. Summary

A blocking + two-stage classifier pipeline with an ensemble of cross-encoders. Candidates come
from three blockers (word TF-IDF, character TF-IDF and a fine-tuned
`Snowflake/snowflake-arctic-embed-m-v2.0` bi-encoder), whose union is pruned by a learned
candidate scorer to ~11.5 candidates per Source-1 record. A LightGBM pair model over 119
hand-built similarity features (stage 1) and fine-tuned `jhu-clsp/mmBERT-base` cross-encoders
score the candidates. A second LightGBM (stage 2) combines those scores with the structure of the
candidate graph (how strongly each target is claimed by other Source-1 records, how similar a
candidate is to its siblings), and one global decision rule tuned on validation macro F0.5 emits
the matches.

The final submission (variant `rawpl2`) is stage 2 over an ensemble of three cross-encoders:

1. **(b)** a reranker that reads normalized text, adapted to the unseen country (France) by one
   self-training round on the pipeline's own unambiguous test decisions;
2. **(c)** a reranker that reads the records as shipped (accents, punctuation, numbering markers),
   trained with a second self-training round;
3. **(d)** a second raw-text reranker trained with a different recipe and seed.

Stage 2 learns on the training countries how much to trust each of them.

**Key ideas:** the learned pruner (F0.5 ceiling 0.9986 at 11.5 candidates per record), the
graph-aware stage 2, self-training that exposes the cross-encoders to French text although no
French labels exist, and the normalized-text / raw-text cross-encoder ensemble.

**Result:** public leaderboard macro F0.5 **0.988**, our highest score (rank 287). Validation
macro F0.5 is 0.99154 for the ensemble's two-reranker base (`rawpl`); the final ensemble's own
validation line was not kept (section 12).

---

## 10. Methodology

### 10.1 Problem analysis

From the data checks (`reports/data_report.md`):

- Training records cover the US and India only; test adds **France** (259,452 of 1,732,544
  Source-1 records, 15 %). No labeled pair crosses countries, and every Source-2/3 record appears
  in at most one Source-1 list.
- **5.58 %** of Source-1 records have no match (the same for the US and India); a matched record
  has **3.67** matches on average. ~26 % of Source-2/3 records belong to no Source-1 record
  (distractors). 13-24 % of India Source-2/3 rows contain Indic scripts; ~3 % of Source-2/3
  records have no address.
- **Name noise:** legal-form changes and drops (`SAS` <-> `SARL`, `Pvt Ltd` <-> `Private Limited`),
  abbreviations and acronyms (`Entraide Sport SARL` -> `ES`), word swaps inside template names,
  inserted tokens (`FRANCE`, `& Associés`, `Groupe`), case, accents added or dropped,
  transliteration.
- **Address noise:** street-type abbreviations (`Rue`/`R.`/`R`, `Road`/`Rd`), house-number formats
  (`Nº 17`, `#26`, `(5)`, `026`, `22bis`), reordered or missing components, landmark references
  (India), and in France a region replaced by its département (`Nouvelle-Aquitaine` <-> `Gironde`).
  Among the France pairs our model matched, about 69 % show that swap and about 12 % of the names
  differ in accents.
- **The hard cases are namesakes:** businesses with the same (template) name on the same street
  with a different house number, and name-only records.
- **The metric** is a macro F0.5 per Source-1 record, singletons included. A wrong emission on a
  singleton costs a full 1.0, and precision weighs twice recall. The pipeline therefore emits only
  confident matches and keeps a correct empty list whenever the evidence is weak.

### 10.2 Solution strategy

**Approach:** blocking + learned candidate pruning + two-stage classifier (hand-feature LightGBM
and an ensemble of mmBERT cross-encoders, combined by a graph-aware LightGBM) + one global
decision rule.

1. **A learned candidate pruner** over the union of three blockers: at most 14 candidates per
   record (11.5 on average) with a far higher recall than fixed per-blocker cut-offs of the same
   budget (validation pair recall 0.9953 vs 0.9857, F0.5 ceiling 0.9986).
2. **Stage 2 sees the whole candidate graph:** for each pair, the best probability any other
   Source-1 record has for the same target, whether this pair is the target's argmax, the pair's
   rank and gap inside its own list, and similarities to sibling candidates. One business appears
   several times across Sources 2 and 3, so a confident match vouches for its near-duplicates,
   and a target strongly claimed elsewhere is not merged twice.
3. **An ensemble of cross-encoders with two views of the same pair:** normalized text
   (`name_ascii | address_norm`) and the raw record (`business_name | business_address`, original
   case, accents and numbering markers). Two raw-text members with different recipes and seeds add
   diversity. The members are not averaged: each contributes its probability and its own list and
   competition features to stage 2, which learns when to trust which member.
4. **Self-training** on the model's own unambiguous test decisions (two rounds): the only way to
   expose the cross-encoders to French text, since no French labels exist.

### 10.3 Candidate generation (blocking)

Blocking runs inside country partitions discovered from the data (no labeled pair crosses
countries); country never enters a model directly, but it stratifies two statistics: per-country
name frequency and per-country IDF (the `*_pc_*` features). Every Source-1 record (train and test) is blocked,
so each target's competing Source-1 records exist on train as they do on test.

- **Normalization** (`src/normalize/`): lower-case ASCII transliteration, legal forms, street
  types and French abbreviations mapped to one spelling (135 hand-written entries in 13 tables,
  `src/normalize/dicts.py`), plus extracted fields: name core (legal form removed), house number,
  postal code, landmarks.
- **B2w:** word TF-IDF over the normalized name core + address core (2^22 hashed features, words in
  > 2 % of documents ignored, each query uses its 8 rarest words), top 20 per record
  (`sparse_dot_topn`).
- **B1:** character 2-4-gram TF-IDF (`char_wb`) over the normalized name, top 5.
- **B4:** `Snowflake/snowflake-arctic-embed-m-v2.0` fine-tuned with CachedMultipleNegativesRanking
  loss on its own 200k-record Source-1 sample (role 3, disjoint from every downstream training
  sample); 256-d Matryoshka embeddings; exact GPU top-30 search. Fine-tuning raised recall@20 by
  +0.009 (US) and +0.108 (India) over the base model.
- **Pool:** the union, ~48 candidates per record, validation pair recall 0.9959.
- **Pruner (stage 0):** a LightGBM over the pool pairs on cheap features (blocker scores and ranks,
  embedding cosine and its margin over the target's best other Source-1 record, name/address
  token-set similarity, IDF overlap, house-number agreement, ranks inside the pool), fitted on the
  role-3 sample. It keeps each record's best 14 (ranks 11-14 only with p >= 0.001) plus its 5
  nearest embedding neighbours (a safety net for an unseen country).
- **Result:** on test, 84,172,807 pool pairs -> **19,944,225** candidates for 1,732,544 records
  (11.5 per record). On validation: pair recall 0.9953 (US 0.9957, India 0.9946),
  macro-F0.5 ceiling 0.9986. Every round and variant shares this candidate set; only the scoring
  models change after it.

### 10.4 Matching models

**Stage 1: 119 features per pair, LightGBM.**
- *Name:* rapidfuzz ratio / token-sort / token-set / partial similarities of the normalized name,
  the name core and the ASCII transliteration; IDF-weighted token Jaccard and the rarest unmatched
  token's IDF; per-country name frequency of the target and of the Source-1 record (namesake
  density); legal-form agreement; acronym / initials match; digit-set agreement; length and
  token-count differences.
- *Address:* token-set / partial similarities of the normalized address and its core;
  IDF-weighted token overlap; number-set Jaccard; house-number and postal-code agreement (and
  whether either side has one); street-word overlap; landmark presence; empty-address flags.
- *Other:* the blockers' scores and ranks, the embedding cosine and its margins (over the
  target's best other Source-1 record and over the record's second-best candidate), the pruner
  probability, and each pair's rank by each similarity inside its Source-1 list.
- *Model:* learning rate 0.1, 63 leaves, feature fraction 0.8, L2 1.0; one fit on the candidate
  pairs of 350,000 Source-1 records (role 1), early-stopped (100 rounds) on a held-out fifth of
  them; best iteration 1,411.

**Cross-encoder rerankers.** `jhu-clsp/mmBERT-base` (MIT, 307M parameters) as a cross-encoder
with one logit: BCE loss, one epoch, warm-up 5 %, weight decay 0.01, bf16, length-grouped
batches. Four models are trained; (b), (c) and (d) form the final ensemble.

- **(a) Main reranker** (round 1; the start point of (b), not in the final ensemble). Reads
  `name_ascii | address_norm` (max length 128, batch 64, lr 3e-5). Trains on the pairs of 150,000
  role-1 records: every true pair plus up to 3 negatives per positive (75 % the hardest by pruner
  probability, 25 % random), 1,455,805 pairs of which 517,277 positive.
- **(b) Self-trained reranker** (round 2; stage-2 prefix `rr_`, score tag `_pl`). Continues (a)
  on the model's own unambiguous test decisions. A record qualifies when every emitted pair has
  stage-2 p >= 0.95 and every other candidate p <= 0.05, or when it emits nothing and its best
  candidate has p <= 0.02. It uses 100,000 qualifying France records (of 134,464) and 15,000 of
  each training country, 2 negatives per positive (1,140,092 pseudo pairs, 446,567 positive), mixed
  with 0.3 replayed training pairs per pseudo pair against forgetting, lr 1e-5.
- **(c) Raw-text reranker** (round 3; prefix `rw_`, tag `_raw`). Reads `business_name |
  business_address` as shipped (max length 96, batch 128, lr 4.2e-5; in a pilot with the same
  6,000 steps and 60k validation pairs, log loss 0.0980 raw vs 0.1121 normalized). One mixed epoch
  on 30 % of the pairs of all 350,000 role-1 records (for time) plus a second self-training round:
  pseudo-labels from round 2's decisions with a looser bar (0.90 / 0.10, empty list 0.05), up to
  120,000 France records and 15,000 per training country, 2 negatives per positive; every pseudo
  pair is kept.
- **(d) Second raw-text reranker** (prefix `rx_`, tag `_raw2`). A diversity member on exactly the
  same training pairs and scored pairs as (c), with a different recipe and seed: seed 43, batch
  128, lr 4.2e-5; stage A is one epoch on 60 % of the train-split pairs (twice (c)'s share),
  stage B continues on the pseudo-labeled pairs plus 0.3 replayed train pairs per pseudo pair at
  lr 1e-5.
- **Which pairs the rerankers score:** every candidate with stage-1 p >= 0.02, each record's 2
  best by stage-1 p and its 3 nearest embedding neighbours (8,139,218 of 19,944,225 test pairs).
  On train they also score every competing pair of the same targets, so the competition features
  are as complete on train as on test. Skipped pairs get missing reranker features, which
  LightGBM handles natively.

**Stage 2 (the ensemble combiner).** LightGBM (learning rate 0.05, 63 leaves, feature fraction
0.8, L2 1.0, early stopping 100 rounds) over the candidate graph, trained on the pairs of 400,000
Source-1 records never seen by stage 1 (role 0; 4.6M pairs). Inputs per pair:
- stage-1 p and its list context: best and second p, rank, ratio and gap to the best, list size,
  sum of p, number of confident candidates;
- target competition: the best p of another Source-1 record for the same target, the margin,
  an argmax flag, the number of competing records;
- sibling similarities: name, address and embedding cosine to the record's two other top
  candidates, and their p;
- 11 stage-1 pair features carried through: embedding cosine and margin, pruner p, name/address
  similarities, IDF overlaps, house-number agreement, empty address, name frequency, rarest
  unmatched target token;
- **for each ensemble member** ((b) `rr_*`, (c) `rw_*`, (d) `rx_*`): its probability, gap to the
  list's best, number of confident candidates in the list, and the target competition computed on
  that member's scores.

### 10.5 Decision rule

One global rule, identical for every country. A record's best candidate is emitted when its
stage-2 p >= `t_emit`; further candidates when p >= `t_add` and p >= `rel` x best. The three values
are chosen by exhaustive grid search on validation macro F0.5 (`decide.grid` in `src/config.yaml`):
`t_emit` in {0.20, 0.25, ..., 0.85} (14 values), `t_add` in {0.20, 0.25, ..., 0.95} (16 values),
`rel` in {0.0, 0.4, 0.6, 0.7, 0.8, 0.9}. Each round and variant re-tunes the rule on its own
stage-2 scores; round 1 chose `t_emit` 0.55, `t_add` 0.75, `rel` 0.7 (`logs/v4/tune.log`), and
the values of `rawpl2` were not kept. The alternative `expected_f` selection (per-record expected
F0.5 on calibrated p) is implemented but not used.

### 10.6 Validation and data roles (leakage control)

Validation is 15 % of the training Source-1 records by seeded hash (330,718 records), never used
to fit any model and never pseudo-labeled. Every model's training sample is disjoint from it and
from the samples of the models downstream of it, so stage 2 and the decision rule see
out-of-sample scores everywhere:

| sample | size | used by | never used by |
| --- | --- | --- | --- |
| validation | 15 % of training S1 (330,718) | rule tuning and reporting only | any model fit, any pseudo-label |
| role 3 | 200,000 S1 | bi-encoder B4 fine-tuning, pruner | stage 1, rerankers, stage 2 |
| role 1 | 350,000 S1 | stage 1; rerankers (a) 150k of them, (c)/(d) all 350k | stage 2 |
| role 0 | 400,000 S1 | stage 2 | stage 1, rerankers, B4, pruner |
| test, confident decisions | capped per country (10.4) | self-training of (b), (c), (d) | stage 1, stage 2, rule tuning |

All remaining training S1 are still blocked and scored, so the competition features (other S1
claiming the same target) are as complete on train as on test.

---

## 11. Results and error analysis

| model | validation macro F0.5 | public LB |
| --- | --- | --- |
| round 1: main pipeline, reranker (a) | 0.99078 (US 0.99023, India 0.99161) | 0.984 |
| round 2: self-trained reranker (b) | 0.99075 | 0.986 |
| round 3 `raw`: raw-text reranker (c) alone | 0.99145 | - |
| round 3 `rawpl`: (b) + (c) | 0.99154 (US 0.99099, India 0.99236) | 0.988 |
| **`rawpl2` (final): (b) + (c) + (d)** | **not kept** | **0.988** |

- Validation cannot measure France. Assuming the US and India score on test as on validation,
  France is implied at about 0.95 for round 1 and 0.97 for `rawpl`: the whole gap between
  validation and the leaderboard.
- Round 1 in detail: singleton accuracy 0.994, pair precision 0.9988, recall 0.9741. On
  validation only 1,389 false pairs remain against 24,236 missed true pairs: the tuned rule is
  precision-heavy, as F0.5 rewards.
- **Common false positives (wrong merges):** namesakes: the same template name on the same street
  with a different house number (`Amicale de Nationale, 30 Rue Cardinal Feltin` vs `... SNC, # 37
  Rue Cardinal Feltin`), or the same address with one changed name word.
- **Common false negatives (missed matches):** name-only targets (no address), acronyms and heavily
  rewritten names (`Hu Centre EI` -> `HC`), and records whose address components were reordered or
  replaced by a broader region.
- **Self-training (round 2)** recovered some of these on France: it rewrote 10.9 % of France lists
  (0.8 % for the US and India), adding name-only and acronym matches and removing namesake merges.
  The leaderboard rose by 0.002 while validation stayed flat (0.99075 vs 0.99078), as expected
  without French validation records.
- **The raw-text reranker (round 3)** gives the largest validation gain after round 1 (`rawpl`
  vs round 1: +0.00076).
- **What the ensemble changes:** relative to `rawpl`, member (d) changes 14,559 of 1,732,544 test
  lists (0.84 %): it adds 10,504 pairs and removes 4,359, turns 378 empty lists into matches and 204
  matched lists into empty ones (section 12).
- **What did not help:**
  - a street-name veto (drop a match whose two street names clearly disagree): it removed ~8k
    France matches, mostly true ones, and left the leaderboard at 0.984; on this data a
    business's records legitimately differ in street;
  - per-country IDF statistics instead of global ones (leaderboard 0.976 vs 0.977); the final
    stage 1 keeps both (global and `*_pc_*`) as features;
  - monotone constraints on stage 1 (-0.005 on validation);
  - exclusivity on stage-2 p (each target kept only for its best Source-1 record): neutral, it
    changes 5 lists;
  - the per-pair probabilities of a second pipeline of ours as extra stage-2 inputs (`rawplx`:
    validation 0.99153 vs 0.99154, leaderboard 0.987 vs 0.988), and intersecting the two
    pipelines' outputs.
- The design experiments behind these choices are in `experiment_log.md`.

---

## 12. The final ensemble (`rawpl2`)

**Members.**

| member | stage-2 prefix | score file | reads | training | seed |
| --- | --- | --- | --- | --- | --- |
| (b) self-trained reranker | `rr_` | `scores_<split>_pl.parquet` | `name_ascii \| address_norm` | (a) on 150k role-1 S1, then round-2 pseudo-labels + replay, lr 1e-5 | 42 |
| (c) raw-text reranker | `rw_` | `scores_<split>_raw.parquet` | `business_name \| business_address` | one mixed epoch: 30 % of the role-1 pairs (350k S1) + round-3 pseudo-labels | 42 |
| (d) second raw-text reranker | `rx_` | `scores_<split>_raw2.parquet` | `business_name \| business_address` | stage A: 60 % of the role-1 pairs; stage B: round-3 pseudo-labels + 0.3 replay, lr 1e-5 | 43 |

**Why these members.**
- (b) and (c) see different text: normalization makes spelling variants equal, while raw text
  keeps accents, `Nº`, `#`, `R.` and case, which carry evidence in France.
- (c) and (d) see the same text and the same pseudo-labels but differ in recipe (mixed epoch vs
  two stages), in the share of training pairs (30 % vs 60 %) and in seed, so their errors are only
  partly correlated.
- All members score exactly the same pairs (`handoff_pairs_<split>.parquet`), so stage 2 compares
  them pair by pair.

**How they are combined.** `stack.rerank_blocks = [["rr", "_pl"], ["rw", "_raw"], ["rx", "_raw2"]]`
(`RAW_VARIANTS["rawpl2"]` in `src/run_pipeline.py`). For each member, stage 2 receives the pair's
probability, its gap to the best probability in the Source-1 list, the number of confident
candidates in the list, and the target competition computed on that member's scores. Stage 2 is
retrained on role-0 records with all three blocks, and the decision rule is re-tuned on
validation. There is no fixed averaging weight: LightGBM learns on the US and India when each
member is reliable.

**Validation log.** The `finish` step that built `rawpl2` ran on the machine holding member (d)'s
scores, and its `tuned decision` line was not copied before that machine was released. So only
the leaderboard score of `rawpl2` is known; a re-run prints it in `logs/v4/raw/raw_finish.log`.

**Test output of the final file** (`output/matching_results.tsv`, compared with `rawpl`):

| | `rawpl` | `rawpl2` (final) |
| --- | --- | --- |
| rows (Source-1 records) | 1,732,544 | 1,732,544 |
| emitted pairs | 5,856,258 | 5,862,403 |
| records with at least one match | 1,633,249 | 1,633,423 |
| empty lists | 99,295 (5.73 %) | 99,121 (5.72 %) |
| matches per matched record | 3.586 | 3.589 |
| lists that differ from `rawpl` | - | 14,559 (0.84 %) |
| pairs added / removed vs `rawpl` | - | +10,504 / -4,359 |
| empty -> matched / matched -> empty | - | 378 / 204 |

The empty share (5.72 %) is close to the 5.58 % singleton rate of the training data, so the
ensemble does not merge singletons at scale. The file has one row per test Source-1 record, no
duplicate rows, no duplicate IDs within a list and only `S2-`/`S3-` IDs. Stage 2 only scores
candidate pairs, so every emitted pair was a candidate in the run that produced it; against the
regenerated `candidate_pairs.tsv`, 631 of the 5,862,403 emitted pairs are missing (section 8).

---

## 13. Variants and the majority vote

`bash run_v4.sh raw` (step `finish`) builds every stage-2 variant whose inputs exist. They share
the candidate set and differ in which cross-encoders stage 2 reads (`RAW_VARIANTS` in
`src/run_pipeline.py`):

| variant | stage 2 reads | validation F0.5 | public LB |
| --- | --- | --- | --- |
| round 1 (`output/`) | main reranker (a) | 0.99078 | 0.984 |
| `pseudo_label` (round 2) | self-trained reranker (b) | 0.99075 | 0.986 |
| `raw` | raw-text reranker (c) alone | 0.99145 | - |
| `rawpl` | (b) + (c) | 0.99154 | 0.988 |
| `rawboth` | (a) + (c) | 0.99154 | - |
| `pla`, `rawpla` | `pseudo_label`, `rawpl` + exclusivity (a target goes only to its best Source-1 record) | 0.99075, 0.99154 | - |
| **`rawpl2` (final)** | (b) + (c) + second raw-text reranker (d) | not kept | **0.988** |

- Any variant becomes the final file with `BER_FINAL_VARIANT=<variant> bash run_final.sh`.
- Variants with an `x` in the name (`plx`, `rawplx`, `rawplxa`, `rawpl2xa`) read per-pair
  probabilities from a second pipeline of ours that is not part of this repository
  (`raw.extra_pairs`). They are skipped automatically (`inputs missing`); the final variant does
  not use them.

**Majority vote (alternative, not submitted).** `tools/vote.py` keeps a target for a Source-1
record when at least `--min` of the given files emit it; `BER_FINAL_VARIANT=vote bash
run_final.sh` runs it on `rawpl`, `rawpl2` and `pseudo_label`:

```bash
cd code/business_entity_resolution
python tools/vote.py --min 2 -o ../../output/matching_results.tsv \
    ../../output/rawpl/matching_results.tsv \
    ../../output/rawpl2/matching_results.tsv \
    ../../output/pseudo_label/matching_results.tsv
```

The candidate set is unchanged (every voted pair was a candidate). With these three inputs the
vote emits 5,858,389 pairs, between the intersection (5,851,899) and the union (5,866,762) of
`rawpl` and `rawpl2`, and it differs from `rawpl` on 5,752 lists. A vote only removes matches
that a single model supports, while stage 2 already weighs the members pair by pair with the
graph context, so the vote stayed an alternative; its leaderboard score was not recorded.

---

## 14. Configuration and troubleshooting

- **Environment variables:** `BER_DATA_DIR` (data, default `dataset/`), `BER_OUTPUT_DIR` (default
  `output/`), `BER_WORK_DIR` (caches and models, default this folder), `BER_LOG_DIR` (runner logs,
  default `logs/v4/`), `BER_CONFIG` (another config file).
- **Driver older than 580:** after the install, switch to the CUDA 12.6 build:
  `uv pip install --python .venv torch==2.14.0 --index-url https://download.pytorch.org/whl/cu126`
  (torch 2.14 has no cu128 build).
- **Faster inference on a big GPU:** `embed.encode_batch` and `rerank.score_batch` (inference
  batch sizes) can be raised, which changes speed and floating-point noise only. Keep the
  training batch sizes (`embed.batch_size`, `embed.mini_batch_size`, `rerank.batch_size`,
  `raw.rerank.batch_size`, `raw.second.batch_size`): they define the models.
- **CUDA out of memory:** lower `embed.encode_batch` (encoding) or `rerank.score_batch` (scoring),
  then re-run the same command.
- **Out of RAM:** lower `part_procs` and `workers`.
- **`numpy max 2.2.6` or other resolver errors:** the venv is not Python 3.12.
- Keep `transformers` at 4.57.6: the arctic-embed model's custom code breaks on 5.x.
- The warning `tokenizer ... with an incorrect regex pattern` when loading mmBERT is harmless (it
  also appears in the runs that produced the submission).
- **Recompute a step:** delete its cache (e.g. `models/reranker_raw2/`, `artifacts/candidates/test_pruned/`)
  and re-run. `run_v4.sh` deletes its `logs/.../.done_*` markers at start, so skipping always
  comes from the cached files.
- `run_v4.sh raw` first times 200 training steps; if the projected training time exceeds
  `BER_RAW_MAX_TRAIN_MIN` (default 200 min), it trains round 3 on only 60 % of its usual share of
  training pairs. Our run used the full share (30 %, `raw.train_frac`). To rule the fallback out on
  a slower GPU, set `BER_RAW_MAX_TRAIN_MIN=100000`.
- Hugging Face models are downloaded on first use and cached; set `HF_HUB_OFFLINE=1` afterwards to
  run without network.

---

## 15. Code map

| stage | module | command (`python -m src.run_pipeline ...`) |
| --- | --- | --- |
| data checks, raw parquet caches, entity keys, disjoint samples | `src/eda.py`, `src/io.py` | `--stage eda` |
| normalization (hand-written maps in `dicts.py`) | `src/normalize/` | `--stage normalize` |
| lexical blockers (word / char TF-IDF) | `src/blocking.py` | `--stage block --step lexical --split all` |
| embedding blocker (fine-tuned arctic-embed) | `src/embed_finetune.py` | `--stage embed --step train`, `--step index --split all` |
| pool union | `src/blocking.py` | `--stage block --split all` |
| candidate pruner: the candidate set | `src/prune.py` | `--stage prune --step fit`, `--step select --split all` |
| 119 pair features | `src/features.py` | `--stage features --split all` |
| stage 1 (LightGBM) | `src/train.py` | `--stage train` |
| mmBERT cross-encoders (a)-(d), pseudo-labels, handoff | `src/rerank.py` | `--stage rerank ...`, `--stage pseudo`, `--stage raw ...` |
| stage 2 over the candidate graph | `src/stack.py` | `--stage stack` |
| decision rule, output files, validation | `src/decide.py` | `--stage tune`, `--stage predict` |
| rounds 2-3 and the stage-2 variants | `src/run_pipeline.py` (`pseudo_config`, `raw_config`, `RAW_VARIANTS`) | `--stage pseudo`, `--stage raw --step data\|time\|train\|score\|handoff\|train2\|score2\|finish` |
| every setting | `src/config.yaml` | |

Development tools (not needed to reproduce the submission): `src/analysis.py` (validation error
analysis), `src/experiment.py` + `src/exp_table.py` (quick stage-1 experiments; `src/train.py`
reuses `experiment.py`'s cached fit), `src/stress.py` (stress-test set, `--stage stress`),
`tools/diag_unseen_empty.py` and `tools/variant_unseen_rate.py` (France diagnostics),
`tools/make_synthetic.py` (synthetic data for `tools/smoke_v4.py`), `tools/vote.py` (section 13).

---

## 16. Fair play and licenses

- Only the provided challenge files are used: no external data, APIs, geocoders or business
  registries.
- The normalization maps (legal forms, street types, French abbreviations; 135 entries in 13
  tables) are hand-written in `src/normalize/dicts.py`.
- Every model is permissively licensed and far below 8B parameters.
- **Disclosure:** the self-training rounds train the cross-encoders (b), (c) and (d) further on the
  pipeline's own unambiguous decisions for test records. No labels and no external information
  are involved. The number of such records is capped per country, with a larger cap for countries
  absent from training. Besides this cap and the blocking partitions, the country label only
  stratifies two stage-1 statistics (per-country name frequency and per-country IDF).

| model | role | license | parameters |
| --- | --- | --- | --- |
| `Snowflake/snowflake-arctic-embed-m-v2.0` | bi-encoder blocker B4 (fine-tuned) | Apache-2.0 | ~305M |
| `jhu-clsp/mmBERT-base` | cross-encoders (a)-(d), each fine-tuned separately | MIT | 307M |
| LightGBM | pruner, stage 1, stage 2 | MIT | tree models |

---

## 17. Conclusion

Learned candidate pruning, a graph-aware second stage and an ensemble of complementary
cross-encoders (normalized and raw text, two raw-text members with different recipes, adapted to
the unseen country by self-training on the model's own unambiguous decisions) reach 0.9915 macro
F0.5 on validation and 0.988 on the public leaderboard. On this task the remaining errors are
decisions among near-identical namesakes and name-only records. There, the candidate graph and
models that have actually seen the target country's text matter more than any single similarity
feature, and combining several such models inside the graph-aware stage 2, instead of averaging
them, lets the combiner learn when each view is reliable.
