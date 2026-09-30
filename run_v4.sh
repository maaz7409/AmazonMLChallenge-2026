#!/usr/bin/env bash
# Runner for rounds 1-3 of the pipeline on one GPU machine, GPU and CPU work overlapped;
# resumable (re-run the same command after any crash or reboot: every finished step is
# skipped by the pipeline's caches). Details: code/business_entity_resolution/README.md.
#
#   bash run_v4.sh            # round 1, main run -> output/matching_results.tsv + candidate_pairs.tsv
#   bash run_v4.sh pseudo     # round 2, pseudo-label round (needs round 1) -> output/pseudo_label/
#   bash run_v4.sh v41        # round 1 (finished steps skip), then round 2
#   bash run_v4.sh raw        # round 3, raw-text reranker (needs rounds 1-2) -> output/<variant>/;
#                             #   logs in logs/v4/raw/, so it can run next to a running round 2
#
# The final variant rawpl2 also needs ensemble member (d) (README.md, section "Reproduce").
#
# Logs: logs/v4/<step>.log (one per step) and logs/v4/progress.log (step starts/ends).
# Watch:  tail -f logs/v4/progress.log
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "$ROOT/.venv/bin/activate" ]; then
  source "$ROOT/.venv/bin/activate"
fi
cd "$ROOT/code/business_entity_resolution"
PY="${PYTHON:-python}"
LOG="${BER_LOG_DIR:-$ROOT/logs/v4}"
if [ "${1:-}" = "raw" ]; then LOG="$LOG/raw"; fi   # own step markers: never touches another run's
mkdir -p "$LOG"
rm -f "$LOG"/.done_* "$LOG"/.failed

say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG/progress.log"; }
step() {  # step <name> <run_pipeline args...>: one stage, its own log; marks done or failed
  local name="$1"; shift
  [ -f "$LOG/.failed" ] && exit 1
  say "start $name"
  if "$PY" -m src.run_pipeline "$@" >> "$LOG/$name.log" 2>&1; then
    touch "$LOG/.done_$name"
    say "done  $name ($(grep -o 'finished in [0-9.]*s' "$LOG/$name.log" | tail -1))"
  else
    touch "$LOG/.failed"
    say "FAILED $name: see $LOG/$name.log"; tail -30 "$LOG/$name.log"; exit 1
  fi
}
wait_for() {  # block until another chain has finished step $1 (or anything failed)
  while [ ! -f "$LOG/.done_$1" ]; do
    [ -f "$LOG/.failed" ] && exit 1
    sleep 15
  done
}

if [ "${1:-}" = "pseudo" ]; then
  say "round 2: pseudo-label round (needs a finished round 1)"
  step pseudo --stage pseudo
  say "pseudo-label outputs: output/pseudo_label/ (validation F0.5: see logs/v4/pseudo.log, 'tuned decision')"
  exit 0
fi

if [ "${1:-}" = "raw" ]; then
  say "round 3: raw-text reranker (needs finished rounds 1 and 2)"
  step raw_data --stage raw --step data
  if [ ! -f models/reranker_raw/config.json ] && [ ! -f "$LOG/.frac" ]; then   # not yet trained, no fraction chosen
    step raw_time --stage raw --step time        # 200 steps: projected full training time
    proj=$({ grep -o "'projected_full_minutes': [0-9.]*" "$LOG/raw_time.log" || true; } | tail -1 | grep -o '[0-9.]*$')
    say "projected full training: ${proj:-?} min"
    if [ -n "$proj" ] && [ "${proj%.*}" -gt "${BER_RAW_MAX_TRAIN_MIN:-200}" ]; then
      echo 0.6 > "$LOG/.frac"                     # too slow: 60% of the usual train_frac (0.3), every pseudo pair kept
      say "training would take ${proj%.*} min > ${BER_RAW_MAX_TRAIN_MIN:-200}: using 60% of the train pairs"
    fi
  fi
  if [ -f "$LOG/.frac" ]; then export BER_RAW_TRAIN_FRAC=$(cat "$LOG/.frac"); fi
  step raw_train --stage raw --step train
  step raw_score --stage raw --step score --split all
  step raw_finish --stage raw --step finish
  say "raw-text outputs: output/<variant>/, e.g. output/raw/, output/rawpl/ (see code/business_entity_resolution/README.md)"
  { grep -o 'tuned decision.*' "$LOG/raw_finish.log" || true; } | while read -r line; do say "validation: $line"; done
  exit 0
fi

say "v4 run on $(getconf _NPROCESSORS_ONLN) CPUs, GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo none)"
step eda --stage eda
step normalize --stage normalize

cpu_chain() {
  step lexical_train --stage block --step lexical --split train
  step lexical_test --stage block --step lexical --split test
  wait_for index_train
  step union_train --stage block --split train
  step prune_fit --stage prune --step fit
  step select_train --stage prune --step select --split train
  step features_train --stage features --split train
  step stage1_fit --stage train --step fit
  wait_for index_test
  step union_test --stage block --split test
  step select_test --stage prune --step select --split test
  step features_test --stage features --split test
  step stage1_test --stage train --step test
}
gpu_chain() {
  step embed_train --stage embed --step train
  step index_train --stage embed --step index --split train
  step index_test --stage embed --step index --split test
  wait_for select_train
  step rerank_train --stage rerank --step train
  wait_for stage1_fit
  step rerank_score_train --stage rerank --step score --split train
  wait_for stage1_test
  step rerank_score_test --stage rerank --step score --split test
}

( cpu_chain ) & CPU_PID=$!
( gpu_chain ) & GPU_PID=$!
fail=0
wait $CPU_PID || fail=1
wait $GPU_PID || fail=1
if [ $fail -ne 0 ] || [ -f "$LOG/.failed" ]; then say "a chain failed; stopping (re-run to resume)"; exit 1; fi

step stage2 --stage stack
step tune --stage tune
step predict --stage predict
say "all done: output/matching_results.tsv + output/candidate_pairs.tsv (validated)"
say "validation F0.5: $(grep -o 'tuned decision.*' "$LOG/tune.log" | tail -1)"
if [ "${1:-}" = "v41" ]; then
  step pseudo --stage pseudo
  say "pseudo-label outputs: output/pseudo_label/"
  say "validation F0.5 (pseudo-label round): $(grep -o 'tuned decision.*' "$LOG/pseudo.log" | tail -1)"
fi
