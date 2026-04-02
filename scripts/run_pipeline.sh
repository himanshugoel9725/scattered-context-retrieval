#!/usr/bin/env bash
# Run remaining experiments sequentially after exp2_1 finishes.
# Usage: bash scripts/run_pipeline.sh [--skip-wait]
# The --skip-wait flag lets you start from exp2_2 directly (if exp2_1 is already done).

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=".venv/bin/python"
LOGS="results/logs"

log() { echo "[pipeline] $(date '+%H:%M:%S') $*"; }

# ──────────────────────────────────────────────────────────────────────────────
# 0. Wait for exp2_1 to complete (checkpoint deleted + results.json fresh)
# ──────────────────────────────────────────────────────────────────────────────
if [[ "${1:-}" != "--skip-wait" ]]; then
    CHECKPOINT="results/exp2_1/exp2_1_checkpoint.json"
    RESULTS="results/exp2_1/exp2_1_results.json"
    log "Waiting for exp2_1 to finish (polling every 60s)..."
    while true; do
        # exp2_1 is done when checkpoint is gone and results.json exists
        if [[ ! -f "$CHECKPOINT" && -f "$RESULTS" ]]; then
            # Extra check: results.json must be newer than rerun2.log start
            log "exp2_1 results.json found and checkpoint gone — proceeding."
            break
        fi
        sleep 60
    done
fi

# ──────────────────────────────────────────────────────────────────────────────
# 1. Exp 2.2 — ablation study (ScatterQA, gpt-4.1-nano, ~20 min)
# ──────────────────────────────────────────────────────────────────────────────
log "Starting exp2_2 ablation..."
PYTHONPATH=. "$PYTHON" experiments/phase2/exp2_2_ablation.py \
    2>&1 | tee "$LOGS/exp2_2_rerun2.log"
log "exp2_2 done."

# ──────────────────────────────────────────────────────────────────────────────
# 2b. Wait for the NEW exp2_1 run to finish before running ICS/judge/exp3_3
#     (exp2_2 doesn't depend on exp2_1 results, but ICS/judge/exp3_3 do)
# ──────────────────────────────────────────────────────────────────────────────
EXP2_1_LOG="results/logs/exp2_1_rerun2.log"
if ! grep -q "Exp 2.1 complete" "$EXP2_1_LOG" 2>/dev/null; then
    log "Waiting for new exp2_1 run to finish (polling every 60s)..."
    while ! grep -q "Exp 2.1 complete" "$EXP2_1_LOG" 2>/dev/null; do
        sleep 60
    done
    log "exp2_1 new run finished — proceeding to ICS/judge/exp3_3."
fi

# ──────────────────────────────────────────────────────────────────────────────
# 2. ICS eval (ScatterQA × 6 strategies, gpt-4.1-nano, ~15 min)
# ──────────────────────────────────────────────────────────────────────────────
log "Starting ICS eval..."
PYTHONPATH=. "$PYTHON" scripts/run_ics_eval.py \
    2>&1 | tee "$LOGS/ics_eval.log"
log "ICS eval done."

# ──────────────────────────────────────────────────────────────────────────────
# 3. LLM judge eval (all strategies × all datasets, gpt-4.1-nano)
# ──────────────────────────────────────────────────────────────────────────────
log "Starting batch_judge..."
PYTHONPATH=. "$PYTHON" scripts/run_llm_judge_eval.py \
    2>&1 | tee "$LOGS/batch_judge.log"
log "batch_judge done."

# ──────────────────────────────────────────────────────────────────────────────
# 4. Exp 3.3 — error analysis (~50 LLM calls, fast)
# ──────────────────────────────────────────────────────────────────────────────
log "Starting exp3_3 error analysis..."
PYTHONPATH=. "$PYTHON" experiments/phase3/exp3_3_error_analysis.py \
    2>&1 | tee "$LOGS/exp3_3_run.log"
log "exp3_3 done."

log "ALL DONE. Results:"
log "  exp2_2:      results/exp2_2/exp2_2_results.json"
log "  ICS:         results/exp2_1/exp2_1_ics_scores.json"
log "  judge:       results/exp2_1/exp2_1_judge_scores.json"
log "  exp3_3:      results/exp3_3/exp3_3_results.json"
