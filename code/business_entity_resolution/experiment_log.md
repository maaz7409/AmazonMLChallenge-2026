# Experiment log

Results of the final pipeline's runs (rounds 1-3 and the `rawpl2` ensemble) and of the design
experiments behind its main choices. Validation = 15 % of the training S1 IDs held out by seeded
hash (330,718 S1); LOCO = train on one country, validate on the other (the best available proxy
for France, which has no labels). Macro F0.5 differences are sometimes quoted in points
(1 point = 0.01).

## Runs of the final pipeline (27 September 2026, one L4, 256 vCPU)

### Round 1: main pipeline

- Candidate set: pool of ~48 per S1 (validation pair recall 0.9959) pruned to 11.5 per S1:
  validation pair recall 0.9953 (US 0.9957, India 0.9946), F0.5 ceiling 0.9986; on test
  84,172,807 pool pairs -> 19,944,225 candidates (`logs/v4/select_train.log`, `select_test.log`).
- Stage 1: best iteration 1,411; top features `prune_p`, `emb_margin_vs_other_s1`,
  `emb_is_t_argmax` (`logs/v4/stage1_fit.log`).
- Reranker (a): 1,455,805 training pairs (517,277 positive) from 150,000 role-1 S1, 22,747 steps
  in 50 min; scores 8,139,218 of the 19,944,225 test pairs (`logs/v4/rerank_train.log`,
  `predict.log`).
- Stage 2 on 4,605,648 role-0 pairs; rule t_emit 0.55 / t_add 0.75 / rel 0.7.
- **Validation 0.99078** (US 0.99023, India 0.99161), singleton accuracy 0.9940, pair precision
  0.9988, recall 0.9741; 1,389 false pairs emitted vs 24,236 true pairs missed. **LB 0.984.**

`logs/v4/tune.log` also holds the line of the first run of this pipeline (validation 0.99050,
LB 0.984). Round 1's settings differ from it in four ways: on train the reranker also scores
every pair competing for the same targets (the first run scored 3.4M of 25.4M train pairs vs
8.1M of 19.9M test pairs, so the reranker competition features saw ~1/3 of the competition on
train/validation), two more reranker competition features, stage-2 learning rate 0.05 instead of
0.1, and a wider rule grid.

### Round 2: self-training (pseudo-labels), reranker (b)

- Pseudo-labels from round 1's test decisions: 134,464 of 259,452 France S1 qualify (100,000
  used), 15,000 used per training country; 1,140,092 pseudo pairs (446,567 positive)
  (`logs/v4/pseudo.log`).
- **Validation 0.99075** (flat, as expected: validation has no French records). **LB 0.986.**
- What it changed on France (read from the two outputs; no labels): 10.9 % of France lists (US
  and India 0.8 %). 20,906 pairs removed (33 % same street / different house number, 33 %
  different street, 24 % same address / different name word) and 10,409 added (64 % same address
  with a noised name, 26 % name-only targets, 4 % acronyms).
- France-specific noise the training countries lack, seen in these outputs: region <->
  département swaps in ~69 % of matched pairs, accents added or dropped in ~12 % of matched
  names, `Nº 17` / `#26` / `(5)` / `R.`.
- A street-name veto on round 1's output (drop a match whose street names clearly disagree)
  removed ~8k France matches (8,270 of 883,514), mostly true ones, and did not improve the
  leaderboard (0.984 with and without it); dropped.

### Round 3: raw-text reranker (c) and the stage-2 variants

Recipe, lighter than planned for the deadline: max length 96, batch 128, 30 % of the train
pairs, up to 120k France pseudo S1 from round 2's decisions (bar 0.90 / 0.10), the round-1 pair
selection, no both-order scoring. Raw text costs ~2x the tokens of normalized text; the full
recipe projected ~775 min of training.

| variant | stage 2 reads | validation macro F0.5 (US / India) | LB |
| --- | --- | --- | --- |
| raw | raw reranker (c) alone | 0.99145 (0.9909 / 0.99227) | - |
| **rawpl** | (b) + (c) | **0.99154** (0.99099 / 0.99236) | **0.988** |
| rawboth | (a) + (c) | 0.99154 | - |
| plx / pla | (b) + a second pipeline's probabilities / + exclusivity | 0.99078 / 0.99075 | - |
| rawplx / rawpla / rawplxa | rawpl + a second pipeline's probabilities / + exclusivity / both | 0.99153 / 0.99154 / 0.99153 | 0.987 / - / - |

The raw reranker gives the largest validation gain after round 1 (+0.00076); a second
pipeline's probabilities and exclusivity are neutral. LB 0.986 -> 0.988: validation explains
about a third, France (implied ~0.969) the rest.

### Final ensemble `rawpl2`: (b) + (c) + (d)

Member (d): a second raw-text reranker on the same pairs as (c), two-stage recipe (60 % of the
train pairs, then the pseudo pairs + 0.3 replay at lr 1e-5), seed 43, trained and scored on a
second L4. **LB 0.988.** Its validation line was not copied from that machine. Against `rawpl`
it changes 14,559 of 1,732,544 test lists (+10,504 / -4,359 pairs; 378 empty -> matched, 204
matched -> empty).

## Design experiments

Run during development on earlier, larger versions of the pipeline (before the pruner and the
rerankers). They justify choices the final pipeline keeps; numbers are comparable only within
one table.

### Blockers (20k validation S1 sample, country-partitioned)

Pair recall at K per blocker alone; "search" = sparse top-K time per 20k queries.

| blocker | setting | @10 | @20 | @30 | @50 | search | note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B1 char_wb 2-4, name_ascii | max_df 0.002, top_m 20 | 0.277 | 0.331 | 0.359 | 0.396 | 3 s | **used**; names repeat across businesses (~50 % of S1 share their name_core) |
| B1 | max_df 0.01, top_m 40 | 0.463 | 0.531 | 0.565 | 0.605 | 67 s | |
| B2 char 3-gram, name_core + address_core | max_df 0.01, top_m 10 | 0.802 | 0.841 | 0.859 | 0.878 | 41 s | off (little recall on top of the others) |
| B2 | max_df 0.01, top_m 20 | 0.851 | 0.881 | 0.893 | 0.906 | 93 s | too slow for test (~135 min) |
| B2w words, name_core + address_core | max_df 0.005, top_m 16 | 0.894 | 0.921 | 0.932 | 0.943 | 11 s | |
| B2w | max_df 0.02, top_m 8 | 0.909 | 0.933 | 0.943 | 0.953 | 26 s | **used** |
| B2w | max_df 0.02, top_m 16 | 0.913 | 0.936 | 0.946 | 0.956 | 40 s | |
| B3 exact keys k_name / k_house / k_rare2 / k_postal (cap 500) | | | | | 0.495 / 0.598 / 0.563 / 0.001 | | off; 25 / 17 / 35 / 0 pairs per S1; postal codes are rare or absent in the data, so the postal key finds nothing |

### Embedding blocker B4 (arctic-embed-m-v2.0)

Recall@20 on a fixed sample of 20,000 validation S1 against all 10.3M train targets (exact GPU
inner-product search within country, 256-d Matryoshka):

| model | pair recall | US | India | entity recall | US | India |
| --- | --- | --- | --- | --- | --- | --- |
| base | 0.9354 | 0.9754 | 0.8761 | 0.8299 | 0.9156 | 0.7030 |
| fine-tuned (1 epoch, CachedMNRL batch 256) | **0.9843** | 0.9845 | 0.9841 | **0.9507** | 0.9515 | 0.9496 |

Both countries improve (+0.0091 US, +0.1080 India), so B4 uses the fine-tuned model
(`blocking.b4: ft`). Alone at k = 20 it beats the whole lexical union on validation.

### Candidate budget: fixed cut-offs vs the learned pruner

Validation, from the stored blocker ranks of a larger candidate set; "est. F0.5" = the stage-2
probabilities of that pipeline restricted to the listed pairs, rule re-tuned (approximate).

| candidate rule | mean / p99 candidates per S1 | pair recall | F0.5 ceiling | est. F0.5 |
| --- | --- | --- | --- | --- |
| B4@5 | 5 / 5 | 0.9161 | 0.9818 | 0.97387 |
| B4@10 | 10 / 10 | 0.9746 | 0.9921 | 0.98197 |
| B4@20 | 20 / 20 | 0.9841 | 0.9949 | 0.98356 |
| B4@30 | 30 / 30 | 0.9876 | 0.9960 | 0.98412 |
| B4@10 + B2w@3 | 10.8 / 13 | 0.9806 | 0.9950 | 0.98402 |
| B4@10 + B2w@5 + B1@3 | 13.7 / 18 | 0.9857 | 0.9962 | 0.98474 |
| B4@20 + B2w@5 + B1@3 | 23.6 / 28 | 0.9910 | 0.9975 | 0.98516 |
| B4@20 + B2w@10 + B1@5 + exact-name key | 39.1 / 111 | 0.9939 | 0.9982 | 0.98541 |
| **learned pruner (final pipeline)** | **11.5 / 16** | **0.9953** | **0.9986** | - |

The lexical blockers add recall cheaply on top of B4, and a smaller candidate set also
transferred better across countries (stage-1 LOCO rose on a ~14-per-S1 set). The pruner keeps
fewer candidates than any fixed rule above with comparable recall (B4@5 and B4@10 keep fewer, but
at far lower recall), and its recall is higher than all of them.

### Stage-1 settings and features

One GroupKFold split of a 200k-S1 training sample (80 % fit / 20 % early stopping), full
validation set, rule re-tuned per experiment. Reference: fine-tuned B4 in the candidate set, 119
features, learning rate 0.2: validation 0.98356, LOCO US->India 0.96900, India->US 0.97602.

| change | validation | LOCO US->India | LOCO India->US | decision |
| --- | --- | --- | --- | --- |
| without `lambda_l2` (older features, lr 0.1) vs with it | 0.95406 vs 0.96289 | 0.86821 vs 0.87886 | 0.92918 vs 0.94498 | L2 1.0 kept: without it validation log-loss rose after ~110 rounds |
| drop every embedding feature (`emb_*`, `b4_*`) | 0.97869 | 0.90908 | 0.95546 | kept: they carry most of the cross-country transfer (+5.99 / +2.06 points LOCO) |
| drop the name-frequency, per-country IDF, unmatched-token and competition features | 0.97732 | 0.95604 | 0.96622 | kept (+0.62 validation, +1.30 / +0.98 LOCO) |
| monotone constraints (`lgbm.monotone`) | 0.97860 | 0.96863 | 0.97125 | dropped (-0.50 validation) |
| expected-F0.5 selection on calibrated p (`decide.selection: expected_f`) | 0.98351 | - | - | not used (no gain over the rule) |
| exclusivity (`decide.assignment`) | 0.98384 | - | - | +0.03 there; neutral on the final pipeline (changes 5 lists) |
| **stage 2** over the candidate graph | **0.98575** | 0.98555 | 0.98515 | kept (+0.22; approximate stage-2 LOCO improves both directions) |
| learning rate 0.1 instead of 0.2 | 0.98410 | - | - | lr 0.1 kept (+0.05) |
| per-country instead of global IDF features | 0.98335 | - | - | not adopted (LB 0.976 vs 0.977) |

### Raw-text reranker pilot

Same 6,000 training steps, same 60k validation pairs: raw record text AUC 0.98805 / log loss
0.0980 vs normalized text 0.98612 / 0.1121. This led to reranker (c) in round 3.
