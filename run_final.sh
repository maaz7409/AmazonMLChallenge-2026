#!/usr/bin/env bash
# Rounds 1-3 of the pipeline, then the chosen variant's matching_results.tsv -> output/, validated
# (code/business_entity_resolution/README.md has the details of every step):
#
#   bash run_final.sh              # rounds 1-3: ~15-16 h on 1x L4 (24 GB), ~128 GB RAM, ~150 GB disk
#   BER_FINAL_VARIANT=rawpl bash run_final.sh    # another variant (default: rawpl2, the submission)
#
# 1. main pipeline (blocking, pruner, features, stage 1, reranker (a), stage 2, rule, outputs)
# 2. self-training round (reranker (b): the reranker continues on the model's own unambiguous test decisions)
# 3. raw-text reranker round (reranker (c), second self-training round; stage-2 variants)
# 4. the chosen variant's matching_results.tsv -> output/ (candidate_pairs.tsv is shared by every variant)
#
# The default variant rawpl2 needs ensemble member (d) first (README.md: steps handoff, train2,
# score2, finish); without it, step 4 stops because output/rawpl2/ does not exist yet. Rounds
# 1-3 are cached, so running this script again after those steps only copies and validates.
#
# Setup once: dataset/ at the repository root (train/ and test/ as shipped), Python 3.12 venv
# at .venv with code/business_entity_resolution/requirements.txt (code/business_entity_resolution/README.md, section 3).
# Every step resumes after a crash: re-run the same command.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
VARIANT="${BER_FINAL_VARIANT:-rawpl2}"   # output/<variant>/ of the raw round (final submission: rawpl2)
cd "$ROOT"
[ -f .venv/bin/activate ] && source .venv/bin/activate
bash run_v4.sh                     # 1. main run            -> output/matching_results.tsv, candidate_pairs.tsv
bash run_v4.sh pseudo              # 2. self-training round -> output/pseudo_label/
bash run_v4.sh raw                 # 3. raw-text round      -> output/<variant>/ for every variant
if [ "$VARIANT" = "vote" ]; then    # majority vote of rawpl, rawpl2 (needs member (d), see above) and pseudo_label
  (cd code/business_entity_resolution && python tools/vote.py --min 2 -o ../../output/matching_results.tsv ../../output/rawpl/matching_results.tsv ../../output/rawpl2/matching_results.tsv ../../output/pseudo_label/matching_results.tsv)
elif [ "$VARIANT" != "pseudo_label" ] && [ "$VARIANT" != "main" ]; then
  cp "output/$VARIANT/matching_results.tsv" output/matching_results.tsv
elif [ "$VARIANT" = "pseudo_label" ]; then
  cp output/pseudo_label/matching_results.tsv output/matching_results.tsv
fi
# candidate_pairs.tsv is the same for every variant (the candidate set never changes)
python3 utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test
echo "final outputs: output/matching_results.tsv (variant $VARIANT), output/candidate_pairs.tsv"
