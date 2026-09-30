# The Entity Resolution Pipeline, Explained Simply


This document explains the whole pipeline step by step, written so that someone new to machine
learning can follow it. It also covers the ensemble (`rawpl2`), the majority vote (`vote.py`) and
how caching works. The technical reference and the run instructions are in
[`code/business_entity_resolution/README.md`](code/business_entity_resolution/README.md); section 13
below lists every number with its source.

---

## Table of contents

1. [The problem](#1-the-problem)
2. [The big picture](#2-the-big-picture)
3. [Each step in detail](#3-each-step-in-detail)
4. [How the data is split (why nothing cheats)](#4-how-the-data-is-split-why-nothing-cheats)
5. [Rounds 2 and 3: the France problem](#5-rounds-2-and-3-the-france-problem)
6. [The ensemble model (`rawpl2`)](#6-the-ensemble-model-rawpl2)
7. [The majority vote (`vote.py`)](#7-the-majority-vote-votepy)
8. [Caching: what is reused and from where](#8-caching-what-is-reused-and-from-where)
9. [Score history](#9-score-history)
10. [Known limitations](#10-known-limitations)
11. [Glossary](#11-glossary)
12. [Where things live in the code](#12-where-things-live-in-the-code)
13. [Fact sheet (single source of numbers)](#13-fact-sheet-single-source-of-numbers)

---

## 1. The problem

There are three lists of businesses:

- **Source 1 (S1):** the clean reference list, with no duplicates.
- **Source 2 and Source 3 (S2/S3):** messy lists. The same business can appear several times,
  with typos, abbreviations or missing parts.

For each S1 business, find every S2/S3 record that is the same real business. Some S1 businesses
have no match at all. They are called "singletons" (5.58% of S1 records).

```
S1:  "Sharma Traders Pvt Ltd | 12 MG Road, Pune 411001"
S2:  "SHARMA TRADERS PRIVATE LIMITED | #12, M.G. Rd, Pune"      <- same business
S3:  "Sharma Traders | Near SBI ATM, MG Road"                   <- same business
S2:  "Sharma Traders | 14 MG Road, Pune"                        <- a different shop (namesake!)
```

(These records are invented to illustrate the task.)

### Why it is hard

1. **Size.** The test set has 1.7M S1 records and about 10M S2/S3 records. Comparing every S1
   with every target would mean about 17 trillion pairs, which is far too many.
2. **France.** The training data only has the US and India. The test set adds France (15% of S1
   records), and no French example has a label.
3. **Namesakes.** Many businesses share a template-like name on the same street and differ only in
   the house number. Some records have a name and no address.
4. **Noise.** Legal-form changes (`Pvt Ltd` vs `Private Limited`, `SAS` vs `SARL`), abbreviations,
   acronyms (`Entraide Sport SARL` becomes `ES`), accents added or dropped, Indian scripts,
   landmark addresses ("Near SBI ATM"), and French regions swapped for départements
   (`Nouvelle-Aquitaine` vs `Gironde`).

### The metric: F0.5

For each S1 record you compute:

- **Precision:** of the matches you predicted, how many were right?
- **Recall:** of the true matches, how many did you find?

```
F0.5 = (1.25 x Precision x Recall) / (0.25 x Precision + Recall)
```

F0.5 counts precision **twice as much** as recall, so a wrong merge costs more than a missed match.
The score is averaged over all S1 records ("macro"). For a singleton:

- predicting an empty list scores **1.0**
- predicting anything scores **0.0**

So the pipeline is built to **stay silent when it is not sure**.

---

## 2. The big picture

The pipeline works like a funnel: cheap, rough filters first, then slower and more accurate models
on the few pairs that survive.

```
 ~17 trillion possible pairs
        |
        |  1. Normalize text (clean it up)
        |  2. Blocking: 3 fast search methods     -> ~48 candidates per S1   ("the pool")
        |  3. Pruner (small ML model)             -> ~11.5 candidates per S1 (~20M test pairs)
        |  4. Stage 1: LightGBM on 119 similarity features  -> probability per pair
        |  5. Reranker: mmBERT reads both records together   -> a second probability
        |  6. Stage 2: LightGBM that also looks at the neighbours -> final probability
        |  7. Decision rule (thresholds tuned on validation)
        v
 matching_results.tsv   (+ candidate_pairs.tsv = the output of step 3)
```

The pipeline runs in **three rounds**, each one building on the previous, plus an ensemble step:

| round | command | what it adds | public LB |
| --- | --- | --- | --- |
| 1. Main run | `bash run_v4.sh` | steps 1 to 7 | 0.984 |
| 2. Self-training | `bash run_v4.sh pseudo` | the reranker also learns from its own confident test answers (helps France) | 0.986 |
| 3. Raw-text round | `bash run_v4.sh raw` | a second reranker that reads the original, uncleaned text | 0.988 |
| Ensemble | four commands (section 6) | a third reranker; stage 2 combines three | **0.988** |

`bash run_final.sh` runs the three rounds in order and then copies the chosen variant to
`output/` (default `rawpl2`, whose ensemble steps are run by hand first, see section 6).

The submitted file is **`rawpl2`**: stage 2 uses round 2's reranker, the raw-text reranker and a
second raw-text reranker (section 6).

---

## 3. Each step in detail

### Step 1: Normalization (`src/normalize/`)

Different spellings of the same thing are turned into one form, so that simple comparisons work.

| raw | normalized |
| --- | --- |
| `Pvt Ltd`, `Private Limited` | the same legal form |
| `Rd`, `Road` | `road` |
| `Ets` | `etablissements` |
| `St` / `Ste` at the start of an address part | `saint` / `sainte` |
| `Société` | `societe` (accents and Indian scripts transliterated to ASCII) |

Useful parts are also pulled out as separate fields: the house number, the postal code, the name
without its legal form (the "name core") and landmarks.

The mappings are 135 hand-written entries (13 tables) in `src/normalize/dicts.py`. No external
data is used.

### Step 2: Blocking, finding plausible candidates fast (`src/blocking.py`, `src/embed_finetune.py`)

The goal is to find, for each S1, a few dozen targets that *might* match, without comparing
against all 10M. Three different search methods are used, because each one catches mistakes the
others miss.

| blocker | how it works | good at | keeps |
| --- | --- | --- | --- |
| **B2w**: word TF-IDF | Compares shared words, where rare words count more ("Zyxtronics" means more than "Road") | exact word matches | top 20 |
| **B1**: character TF-IDF | Compares small chunks of 2-4 letters of the name | typos (`Sharma` vs `Sharmaa`) | top 5 |
| **B4**: neural embeddings | A model (`Snowflake/snowflake-arctic-embed-m-v2.0`) turns each record into 256 numbers, a kind of meaning fingerprint. Similar businesses get similar fingerprints. It was fine-tuned on training pairs. | abbreviations, reordering, meaning | top 30 |

Details:

- Blocking runs **separately inside each country**, because no labeled match crosses countries.
  The country is never fed to a model directly; it only stratifies two statistics (per-country
  name frequency and per-country word rarity).
- The embedding model was fine-tuned on its own sample of 200k S1 records. In a development test,
  fine-tuning improved recall@20 by +0.009 (US) and +0.108 (India) over the base model.
- Merging the three results gives about **48 candidates per S1**. This is **the pool**, and it
  contains **99.6%** of the true matches (validation).

### Step 3: The pruner (`src/prune.py`)

48 candidates per S1 is still too many for the expensive models later on. A small **LightGBM**
model (see the glossary) looks at cheap signals and keeps the best ones:

- blocker scores and ranks (which blockers found the pair, and at what position)
- embedding similarity, and how much better it is than this target's similarity to any other S1
- name and address word overlap
- whether the house numbers agree

It keeps the best **14 per S1** (ranks 11-14 only if they are not hopeless, p >= 0.001), and it
always keeps each S1's **5 nearest embedding neighbours** as a safety net for France.

- Result: **~11.5 candidates per S1** (19.9M test pairs), still holding **99.5%** of the true
  matches.
- For comparison, simple fixed cut-offs with a similar budget (B4 top 10 + B2w top 5 + B1 top 3,
  ~14 per S1) keep only 98.6%.
- This candidate set is written to **`output/candidate_pairs.tsv`** and **never changes after
  this step**. Every variant and every round shares it.

### Step 4: Stage 1, LightGBM on hand-made features (`src/features.py`, `src/train.py`)

For each candidate pair, **119 numbers** are computed that describe how similar the two records are.

- **Name:** fuzzy string similarity (rapidfuzz ratio, token-sort, token-set, partial), shared rare
  words weighted by rarity (IDF), whether the legal forms agree, whether one name is the initials
  of the other, shared digits, length differences, and **how common the name is** (a common name
  means more namesakes and more danger).
- **Address:** whether the house number and postal code agree, shared street words, shared
  numbers, landmark presence, whether an address is empty.
- **Other:** which blockers found the pair and at what rank, embedding similarity, the pruner's
  probability, and the pair's rank by each similarity inside its S1's candidate list.

**LightGBM** learns from these numbers which combinations mean "same business". For example: "if
the house numbers differ and the name is very common, then less likely". Its output is a
**probability** for each pair.

- Trained on the candidate pairs of 350k S1 records.
- Learning rate 0.1, 63 leaves, early stopping (it stopped at 1,411 trees).

### Step 5: The reranker, a model that reads both records (`src/rerank.py`)

Stage 1 only sees the numbers you designed. The reranker reads the **text itself**.

- Model: **`jhu-clsp/mmBERT-base`**, a multilingual BERT with 307M parameters and an MIT license.
- It is a **cross-encoder**. The input is `"record A [SEP] record B"` and the output is "same
  business?" as a probability.

**Bi-encoder vs cross-encoder: why use both?**

- The blocker B4 is a **bi-encoder**. It makes a fingerprint for each record *separately* and
  then compares the fingerprints. It is fast (you compute 10M fingerprints once) but rough.
- The reranker is a **cross-encoder**. It reads both records *side by side*, like a human
  comparing them. It is much more accurate but far too slow to run on every pair.

So the reranker only scores the pairs that matter (8.1M of the 19.9M test pairs):

- every pair with stage-1 probability >= 0.02
- each S1's 2 best candidates by stage 1
- each S1's 3 nearest candidates by embedding (so it does not only look at what stage 1 already likes)

**Training data** (1.46M pairs):

- all true pairs from 150k S1 records
- up to 3 wrong pairs per true pair: 75% are the *hardest* wrong pairs (the ones that look most
  similar), 25% are random
- this teaches the model the fine differences, such as house number 12 vs 14
- one epoch, batch 64, lr 3e-5, bf16

This main reranker reads **normalized** text: `"{name_ascii} | {address_norm}"`.

### Step 6: Stage 2, using the neighbours (`src/stack.py`)

This is one of the key ideas of the solution. Stages 1 and 5 judge each pair on its own. Stage 2
looks at the **whole network of candidates**:

- **Competition.** If S1-A says target T is 0.8 likely, but S1-B says the *same* T is 0.99
  likely, then T probably belongs to B. In this data each target belongs to at most one S1.
  Features: the best probability any *other* S1 has for this target, the margin to it, whether
  this pair is the target's best, and how many S1 compete for it.
- **Siblings.** One business appears several times in S2/S3. If a candidate is almost identical to
  another candidate you are already sure about, it is probably also a match. Features: name,
  address and embedding similarity to the S1's other top candidates, and their probabilities.
- **Position in the list:** rank, gap to the best candidate, ratio to the best, number of
  confident candidates, list size.
- **Carried-over evidence:** the stage-1 probability, every reranker's probability (with its own
  list and competition features), and 11 raw pair features (embedding cosine, pruner p,
  name/address similarity, house-number agreement, empty address, name frequency, ...).

Stage 2 is another LightGBM (lr 0.05), trained on **400k S1 records that stage 1 never saw**. This
matters: a model's scores on its own training data are too optimistic. Stage 2 has to learn from
realistic scores, like the ones it will see on test.

### Step 7: The decision rule (`src/decide.py`)

Stage 2's probabilities are turned into lists with 3 numbers:

```
emit the best candidate        if p >= t_emit
emit further candidates        if p >= t_add  AND  p >= rel x (best p)
otherwise                      leave the list empty (no match)
```

Every combination on a grid (14 x 16 x 6) is tried on the validation set, and the one with the
best macro F0.5 wins. The rule is **the same for every country**. In round 1 the winner was
`t_emit 0.55, t_add 0.75, rel 0.7`. Each round and variant re-tunes these numbers for its own
model.

Finally the predict step writes both TSV files and runs the official validator
(`utils/validate_submission.py`).

---

## 4. How the data is split (why nothing cheats)

The training S1 records are split into separate groups ("roles"), so every model is trained on
data the models *before* it never saw. Otherwise a later model would learn from over-confident
scores and trust them too much on test.

| group | size | used for |
| --- | --- | --- |
| validation | 15% (330,718 S1, by seeded hash) | measuring the score and tuning the rule; **never** trained on |
| role 3 | 200k S1 | embedding model (B4) and pruner |
| role 1 | 350k S1 | stage 1; the rerankers use 150k (main) or all 350k (raw) of these |
| role 0 | 400k S1 | stage 2 |

Every training S1 is still *blocked* (given candidates), even if no model trains on it. That way
the "competition" features on train look the same as on test, where every S1 competes.

---

## 5. Rounds 2 and 3: the France problem

On validation (US and India only) round 1 scored 0.991, but the leaderboard said 0.984. The gap
is France, which the models had never seen: assuming the US and India score on test as on
validation, France comes out at about 0.95.

### Round 2: self-training with pseudo-labels (`--stage pseudo`)

No French labels exist, so the model makes its own.

1. Take round 1's answers on **test**, and keep only the S1 records where the model is **very
   sure**:
   - every emitted match has stage-2 p >= 0.95 **and** every rejected candidate has p <= 0.05, or
   - the S1 emits nothing and its best candidate has p <= 0.02 (a confident "no match").
2. Treat those answers as labels: **100k French S1** (of 134k that qualified) plus 15k each from
   the US and India.
3. **Continue training** the reranker on them at a low learning rate (1e-5). Old training pairs
   are mixed in (0.3 per pseudo pair) so that it does not forget what it learned.
4. Re-score all pairs with it, retrain stage 2, re-tune the rule.
   Output: `output/pseudo_label/` (score tag `_pl`).

**Analogy:** a student marks their own homework, keeps only the answers they are certain of, and
studies from those. The student cannot learn anything totally new this way, but they get used
to the new kind of question (French text).

**Effect:** it changed 10.9% of France lists (adding name-only and acronym matches, removing
namesake merges). The leaderboard went from **0.984 to 0.986**. Validation stayed flat (0.99075
vs 0.99078), which is what you would expect because validation has no French records.

> **Disclosure:** this trains on the model's own predictions for test records (no human labels, no
> external data). The technical README says so openly.

### Round 3: the raw-text reranker (`--stage raw`)

Normalization throws information away: accents, `Nº 17`, `#`, `R.`, upper and lower case. A
**second mmBERT** reads the records exactly as shipped:
`"{business_name} | {business_address}"`.

- In a pilot it predicted better than the normalized-text version (log loss 0.098 vs 0.112).
- Its training data includes a **second round of pseudo-labels**, taken from *round 2's* answers,
  with a looser bar (0.90 / 0.10) and up to 120k French S1.
- Because of the deadline: 30% of the training pairs (`raw.train_frac: 0.3`), max length 96,
  batch 128, lr 4.2e-5.
- Scores are saved with tag `_raw` (columns `rw_*` in stage 2).

The `finish` step then builds several **stage-2 variants**, each using a different combination of
rerankers:

| variant | stage 2 uses | validation | leaderboard |
| --- | --- | --- | --- |
| `raw` | raw reranker only | 0.99145 | - |
| `rawpl` | round-2 reranker (`rr_*`) + raw reranker (`rw_*`) | 0.99154 | 0.988 |
| `rawboth` | main (round-1) reranker + raw reranker | 0.99154 | - |
| `plx` / `rawplx` | + probabilities from a second pipeline of ours (not included) | 0.99153 (rawplx) | 0.987 (rawplx) |
| `pla` / `rawpla` / `rawplxa` | + "exclusivity": each target goes only to its best S1 | 0.99075 / 0.99154 / 0.99153 | - |
| **`rawpl2` (submitted)** | + a **second** raw reranker (the ensemble, section 6) | not kept | **0.988** |
| `vote` | majority vote of three files (section 7) | - | not recorded |

`rawpl2` was submitted: it is our highest leaderboard score (0.988, tied with `rawpl` at three
decimals).

---

## 6. The ensemble model (`rawpl2`)

### How the models are combined

The "ensemble" here does **not** average model outputs. The models are combined **inside stage
2**: each reranker's probability becomes extra input columns for the stage-2 LightGBM, and stage 2
learns from data when to trust which one.

`rawpl2` = stage 2 reading **three** rerankers:

| prefix | reranker | text |
| --- | --- | --- |
| `rr_*` | round 2's self-trained reranker | normalized |
| `rw_*` | round 3's raw-text reranker | raw |
| `rx_*` | a **second raw-text reranker** | raw |

The second raw reranker is trained differently so that it makes *different* mistakes (diversity
is what makes an ensemble useful):

- another random seed (43 instead of 42)
- a **two-stage recipe**: first one epoch on the training pairs (60% of them), then continued on
  the pseudo-labeled pairs plus 30% replayed training pairs at lr 1e-5
- we trained it on a **separate GPU machine** to save time (one machine works too)

### How to build it

After round 3 (or at least its data step), from `code/business_entity_resolution/`:

```bash
python -m src.run_pipeline --stage raw --step handoff
#   -> artifacts/rerank/handoff_pairs_{train,test}.parquet (the pair lists to score)
python -m src.run_pipeline --stage raw --step train2
python -m src.run_pipeline --stage raw --step score2 --split all
#   -> artifacts/rerank/scores_{train,test}_raw2.parquet
BER_RAW_VARIANTS=rawpl2 python -m src.run_pipeline --stage raw --step finish
#   -> output/rawpl2/matching_results.tsv
```

To use a second machine, copy the two `handoff_pairs` files and
`artifacts/rerank/train_pairs_raw.parquet` there (same dataset, same code), run `train2` and
`score2` there, and copy the two score files back before `finish`.

`run_v4.sh raw` does not run these four steps: until they have run, `finish` logs
`raw variant rawpl2: inputs missing; skipped`.

---

## 7. The majority vote (`vote.py`)

`code/business_entity_resolution/tools/vote.py` is simple. It runs **after** everything else, on
the final TSV files. For each S1 record, it keeps a target only if **at least 2 of the 3 files**
emit it:

```
inputs:  output/rawpl/    output/rawpl2/    output/pseudo_label/
S1-001:  [A, B, C]        [A, B]            [A, B, D]
vote:    [A, B]           (C and D have only 1 vote each -> dropped)
```

It removes matches that only one model believes in. That helps precision, which F0.5 rewards.

Command (the same as `BER_FINAL_VARIANT=vote bash run_final.sh`, run inside
`code/business_entity_resolution/`):

```bash
python tools/vote.py --min 2 -o ../../output/matching_results.tsv \
    ../../output/rawpl/matching_results.tsv \
    ../../output/rawpl2/matching_results.tsv \
    ../../output/pseudo_label/matching_results.tsv
```

The candidate set does not change, because every voted pair was already a candidate.

On our outputs the vote behaves as a 2-of-3 vote should:

| | pairs |
| --- | --- |
| rawpl AND rawpl2 (both agree) | 5,851,899 |
| vote | 5,858,389 |
| rawpl OR rawpl2 (either) | 5,866,762 |

`rawpl` and `rawpl2` differ on only 14,559 of 1,732,544 S1 lists, so the vote changes very
little. It was kept as an alternative and not submitted: stage 2 already weighs the models pair by
pair with the graph context. Its leaderboard score was not recorded.

---

## 8. Caching: what is reused and from where

The pipeline relies heavily on cached results.

1. **Every step saves its results.** Parquet files go under
   `code/business_entity_resolution/artifacts/` and model weights under `models/`. When you re-run,
   any step whose output already exists is skipped. That is why "re-run the same command after a
   crash" works.
   - `run_v4.sh` deletes its `.done_*` marker files at startup, so the skipping comes from the
     **saved files**, not from the markers.
   - To force a step to recompute, delete its folder, for example `models/reranker/` or
     `artifacts/candidates/<split>_pruned/`.
2. **Rounds 2 and 3 are built on round 1's saved outputs.** They reuse the normalized text,
   blocking, the candidate set, features and stage-1 scores, and only train new rerankers and redo
   stage 2 and the rule.
   - `pseudo` cannot run without round 1's files.
   - `raw` cannot run without round 2's files (`artifacts/preds/test_pl.parquet`).
   - That is why round 3 takes ~5-6 hours on top of round 1's ~8 hours instead of repeating them.
3. **Hugging Face models** (`arctic-embed`, `mmBERT`) are downloaded once, then can be used
   offline (`HF_HUB_OFFLINE=1`).
4. **Some cached files come from outside the three-round run:**

   | file | produced by | needed by |
   | --- | --- | --- |
   | `artifacts/rerank/scores_{train,test}_raw2.parquet` | the ensemble steps (`train2`, `score2`; on this or a second machine) | `rawpl2`, `rawpl2xa`, `vote` |
   | `artifacts/extra/{train,test}_pairs_x.parquet` | a second pipeline of ours, **not included** in this repository | `plx`, `rawplx`, `rawplxa`, `rawpl2xa` (skipped without them) |

   The submitted **`rawpl2`** needs the `raw2` scores and nothing from the second pipeline.

---

## 9. Score history

| run | what it adds | validation F0.5 | public LB |
| --- | --- | --- | --- |
| round 1 | the main pipeline (pruner, stage 1, reranker, stage 2) | 0.99078 | 0.984 |
| round 2 | self-training (pseudo-labels) | 0.99075 | 0.986 |
| round 3 `raw` | raw-text reranker alone | 0.99145 | - |
| round 3 `rawpl` | self-trained + raw-text reranker | 0.99154 | 0.988 |
| round 3 `rawplx` | + a second pipeline's probabilities | 0.99153 | 0.987 |
| **`rawpl2`** | **+ second raw-text reranker (submitted)** | not kept | **0.988** |

**What did not help:** a street-name veto (it removed ~8k French matches, mostly correct ones,
and did not raise the leaderboard), per-country IDF statistics (0.976 vs 0.977 on the
leaderboard), monotone constraints on stage 1, and intersecting the outputs of two pipelines.

**Remaining errors** (from validation): wrong merges are mostly namesakes (same template name,
same street, different house number). Missed matches are often name-only targets with no
address, acronyms and heavily rewritten names. The rule is precision-heavy: 1,389 false pairs vs
24,236 missed pairs on validation (round 1).

---

## 10. Known limitations

1. **The ensemble is not automated.** `rawpl2` (and the `vote` option of `run_final.sh`) need the
   four ensemble commands of section 6 after round 3; `run_v4.sh raw` does not run them.
2. **`rawpl2`'s validation score and rule values were not kept.** They were printed on the machine
   that built it; a re-run prints them in `logs/v4/raw/raw_finish.log`.
3. **The shipped candidate file is a regeneration.** The original run's `candidate_pairs.tsv`
   (19,944,225 pairs) was lost; the file that goes with the final matches was rebuilt by re-running
   round 1 up to the pruner (19,835,499 pairs). It is shipped xz-compressed as
   `output/candidate_pairs.tsv.xz` (extract it with `xz -dk`, see the technical README, section 8). 631 of the 5,862,403 final matches (0.011%) are not in
   it, because GPU training is not bit-for-bit deterministic.
4. **Runs from scratch differ slightly** from ours for the same reason (candidate set, matches,
   score).
5. `run_v4.sh raw` trains on fewer pairs if 200 test steps project more than 200 minutes of
   training (`BER_RAW_MAX_TRAIN_MIN`); on a slow GPU, raise the limit to reproduce our recipe.

---

## 11. Glossary

| term | meaning |
| --- | --- |
| **Entity resolution** | Deciding which records refer to the same real-world thing |
| **Blocking** | A cheap first pass that picks a few plausible candidates, so the expensive models don't compare everything with everything |
| **Recall ceiling** | The best recall you could possibly get, given which true matches survived blocking |
| **TF-IDF** | A way to turn text into numbers where rare words count more than common ones |
| **Embedding** | A list of numbers (here 256) that represents the meaning of a text; similar texts get similar lists |
| **Bi-encoder** | Encodes each record separately into an embedding; fast, used for search |
| **Cross-encoder** | Reads two records together and outputs a match score; slow but accurate |
| **Fine-tuning** | Continuing to train a pre-trained model on your own task's data |
| **LightGBM** | A gradient-boosted decision-tree library: hundreds of small trees, each correcting the previous ones' mistakes |
| **Feature** | One number describing a pair, e.g. "name similarity = 0.93" |
| **Validation set** | Labeled data kept aside and never trained on, used to measure the score honestly |
| **Out-of-sample** | Scores a model gives on data it did not train on (realistic, not over-confident) |
| **Stacking (stage 2)** | A second model that uses the first models' outputs as its inputs |
| **Pseudo-labels / self-training** | Using the model's own confident predictions on unlabeled data as extra training labels |
| **Ensemble** | Combining several models so their different mistakes cancel out |
| **Majority vote** | Keeping an answer only if most models agree on it |
| **Threshold** | The probability above which a pair is declared a match |
| **Log loss** | A measure of how good a model's probabilities are (lower is better) |

---

## 12. Where things live in the code

| what | where |
| --- | --- |
| run instructions and technical write-up | `code/business_entity_resolution/README.md` |
| one-command final run | `run_final.sh` |
| round runner (GPU and CPU chains in parallel, resumable) | `run_v4.sh` |
| every setting | `code/business_entity_resolution/src/config.yaml` |
| entry point, rounds and variants | `src/run_pipeline.py` (`pseudo_config`, `raw_config`, `RAW_VARIANTS`) |
| normalization | `src/normalize/normalize.py`, `src/normalize/dicts.py` |
| blocking | `src/blocking.py` (TF-IDF), `src/embed_finetune.py` (B4) |
| pruner | `src/prune.py` |
| pair features | `src/features.py` |
| stage 1 | `src/train.py` |
| rerankers and pseudo-labels | `src/rerank.py` |
| stage 2 | `src/stack.py` |
| decision rule, outputs | `src/decide.py` |
| majority vote | `tools/vote.py` |
| submission validator | `utils/validate_submission.py` |
| run results and design experiments | `code/business_entity_resolution/experiment_log.md` |
| logs | `logs/v4/progress.log`, `logs/v4/<step>.log`, `logs/v4/raw/` |

---

## 13. Fact sheet (single source of numbers)

Every number quoted in this repository's documents comes from this section. When another
document disagrees, **this section wins**; the "source" column says where each value was
verified. Numbers that were never recorded are listed as such and must not be shown as if they
were known. "README" = `code/business_entity_resolution/README.md`; logs are under `logs/v4/`.

### Data and metric

| fact | value | source |
| --- | --- | --- |
| test Source-1 records | 1,732,544 (France 259,452 = 15 %, India 809,986, US 663,106) | `reports/data_report.md` |
| test Source-2/3 records | ~10M (9,969,589 targets) | `reports/data_report.md`, `index_test.log` |
| training countries | US and India only; France appears only in test | `reports/data_report.md` |
| singletons (no match) | 5.58 % of Source-1 records (US 5.58 %, India 5.59 %) | `reports/data_report.md` |
| matches per matched record | 3.67 on average (3.666) | `reports/data_report.md` |
| Source-2/3 records matching nothing | ~26 % | `reports/data_report.md` |
| metric | macro F0.5 per Source-1 record, $F_{0.5} = \frac{1.25\,P\,R}{0.25\,P + R}$; a singleton scores 1 for an empty list, else 0 | `PROBLEM_STATEMENT.md` |

### Candidate generation

| fact | value | source |
| --- | --- | --- |
| normalization maps | **135 entries in 13 tables** | `src/normalize/dicts.py` |
| blockers | B2w word TF-IDF top 20; B1 char 2-4-gram TF-IDF on names top 5; B4 fine-tuned `snowflake-arctic-embed-m-v2.0`, 256-d, top 30 | `src/config.yaml` |
| B4 fine-tuning gain | recall@20 +0.009 US, +0.108 India (development test) | `experiment_log.md` |
| pool | ~48 candidates per record (48.1); validation pair recall 0.9959 | `union_train.log` |
| pruner keep rule | best 14 per record (ranks 11-14 only if p >= 0.001) + the 5 nearest embedding neighbours | `src/config.yaml` |
| test candidates (the run that made `rawpl2`) | 84,172,807 pool pairs -> **19,944,225** kept = **11.51 per record** | `select_test.log` |
| shipped `candidate_pairs.tsv` | regenerated by re-running round 1 up to the pruner, shipped as `.tsv.xz`: **19,835,499** pairs (11.45 per record); 631 of the 5,862,403 final matches (0.011 %) are outside it | README 8 |
| per-country candidate counts on test | **not recorded** | - |
| candidate-set quality | validation **pair recall** 0.9953 (US 0.9957, India 0.9946), macro-F0.5 ceiling 0.9986; fixed cut-offs of a similar budget (B4@10 + B2w@5 + B1@3, 13.7 per record) reach **pair recall** 0.9857 | `select_train.log`, `experiment_log.md` |

### Models

| fact | value | source |
| --- | --- | --- |
| stage 1 | LightGBM, 119 features, 350k training records, lr 0.1, 63 leaves, best iteration 1,411 | `stage1_fit.log`, `src/config.yaml` |
| rerankers | `jhu-clsp/mmBERT-base` (MIT, 307M) cross-encoders: (a) normalized text, 150k records / 1,455,805 pairs; (b) round-2 self-trained; (c) round-3 raw text; (d) second raw-text reranker, seed 43, two-stage recipe | `rerank_train.log`, README 10.4 |
| pairs the rerankers score | 8,139,218 of 19,944,225 test pairs (stage-1 p >= 0.02, 2 best per record, 3 nearest embedding neighbours) | `predict.log` |
| stage 2 | LightGBM over the candidate graph, 400k records never seen by stage 1 (4,605,648 pairs), lr 0.05 | `stage2.log`, `src/config.yaml` |
| data roles | validation 15 % (330,718), role 3 = 200k (embedder + pruner), role 1 = 350k (stage 1, rerankers), role 0 = 400k (stage 2) | `src/config.yaml`, README 10.6 |

### Decision rule

| fact | value | source |
| --- | --- | --- |
| rule | emit the best candidate if p >= t_emit; others if p >= t_add and p >= rel x best; one global rule, grid-searched on validation | `src/decide.py` |
| recorded values | **only round 1's**: t_emit 0.55, t_add 0.75, rel 0.7. Every variant re-tunes its own; the values of the final `rawpl2` were **not kept**. Examples use round 1's values and say so | `tune.log` (last line) |
| side effect of round 1's values | `rel` never binds: rel x best <= 0.7 < t_add | arithmetic |

### Self-training and ensemble

| fact | value | source |
| --- | --- | --- |
| round 2 | bars 0.95 / 0.05 (empty list 0.02); 100k France (of 134,464 qualifying) + 15k per training country; 1,140,092 pseudo pairs; rewrote 10.9 % of France lists; LB 0.984 -> 0.986 | `pseudo.log`, `src/config.yaml`, `experiment_log.md` |
| round 3 | bars 0.90 / 0.10 (empty list 0.05); up to 120k France records; 30 % of the training pairs | `src/config.yaml`, `experiment_log.md` |
| final `rawpl2` vs `rawpl` | 14,559 lists differ; +10,504 / -4,359 pairs; 378 empty -> matched, 204 matched -> empty | computed from the two output files |
| final `rawpl2` file | 5,862,403 pairs; 1,633,423 records matched; 99,121 empty lists (5.72 %); 3.589 matches per matched record | `output/matching_results.tsv` |

### Results

| model | validation F0.5 | public LB |
| --- | --- | --- |
| round 1 | 0.99078 | 0.984 |
| round 2 (self-training) | 0.99075 | 0.986 |
| `rawpl` | 0.99154 | 0.988 |
| **`rawpl2` (final)** | **not kept** | **0.988** (rank 287 on the public leaderboard) |

Did not help: street-name veto (no leaderboard gain; it removed mostly correct French matches),
per-country IDF (0.976 vs 0.977), monotone constraints (-0.005 on validation), intersecting two
pipelines, a second pipeline's probabilities (`rawplx`, LB 0.987).
