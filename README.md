# Amazon ML Challenge 2026 : Our solution 

## Problem : Business Entity Resolution

For every business in a clean reference list (Source 1), find all records of the same business
in two noisy lists (Sources 2 and 3), across the US, India and a test-only country, France.
Scored by macro F0.5 per Source-1 record.

## Result

public leaderboard macro F0.5 **0.988**, rank **287** of ~25,000 registered teams.

## Approach

Three blockers (word TF-IDF, character TF-IDF, a fine-tuned multilingual
bi-encoder) and a learned pruner keep ~11.5 candidates per record at 99.5 % recall. A LightGBM
over 119 similarity features and three fine-tuned mmBERT cross-encoders score the candidates. A
second, graph-aware LightGBM combines their scores with the competition between records for the
same target, and one global rule tuned on validation F0.5 emits the matches. Two self-training
rounds adapt the cross-encoders to France, which has no labels.

| document | for |
| --- | --- |
| [`code/business_entity_resolution/README.md`](code/business_entity_resolution/README.md) | **running it** (setup, commands, run times) and the full technical write-up |
| [`PIPELINE_EXPLAINED.md`](PIPELINE_EXPLAINED.md) | a beginner-friendly walkthrough, and the fact sheet of every number |
| [`PROBLEM_STATEMENT.md`](PROBLEM_STATEMENT.md) | the challenge task, data format and rules |
| [`code/business_entity_resolution/experiment_log.md`](code/business_entity_resolution/experiment_log.md) | results of every run and the design experiments |
| Visualization site | https://ballin-amazonml2026.vercel.app |

**Quick start** (one GPU with >= 24 GB and bf16, ~128 GB RAM; the challenge data in `dataset/`):

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv -r code/business_entity_resolution/requirements.txt
export CUDA_VISIBLE_DEVICES=0
bash run_v4.sh && bash run_v4.sh pseudo && bash run_v4.sh raw   # rounds 1-3, ~15-16 h on an L4
```

The final ensemble needs four more commands, listed in section 6 of the
[technical README](code/business_entity_resolution/README.md#6-reproduce-the-submission-rawpl2).
The challenge data is not included. The candidate set is shipped compressed as
`output/candidate_pairs.tsv.xz` (to fit on GitHub); extract it with `xz -dk output/candidate_pairs.tsv.xz`
before validating (details in section 8 of the technical README).


## Authors

@maaz7409
@AnshulPatil2005
@KeshavKumar-0
@Nipun-Shekhar