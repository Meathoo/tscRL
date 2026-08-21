#!/usr/bin/env bash
# Resume-on-crash runner, generalised over the simulator world.
#
# This is a separate file rather than a flag on ../resilient_run.sh on purpose:
# that script is executing right now for the long 7x28 jobs, and bash re-reads a
# running script from a byte offset, so editing it in place can corrupt a live
# run. Same reasoning as scripts/ingolstadt_study.sh.
#
# Differences from resilient_run.sh:
#   * --world comes from RESILIENT_WORLD (default cityflow)
#   * the model directory follows the world, matching run.py's output layout
#     (data/output_data/tsc/<world>_<agent>/<network>/<prefix>/model)
#
# Usage (inside the container):
#   RESILIENT_WORLD=sumo RESILIENT_NETWORK=sumo1x21 RESILIENT_EPISODES=250 \
#     scripts/resilient_run_world.sh <prefix> <seed> [extra run.py args...]

set -u
cd "$(dirname "$0")/.."

PREFIX="$1"; shift
SEED="$1"; shift
EXTRA_ARGS=("$@")

WORLD="${RESILIENT_WORLD:-cityflow}"
NETWORK="${RESILIENT_NETWORK:-cityflow4x4}"
AGENT="${RESILIENT_AGENT:-hyperlight_mappo}"
EPISODES="${RESILIENT_EPISODES:-250}"
MODEL_DIR="data/output_data/tsc/${WORLD}_${AGENT}/${NETWORK}/${PREFIX}/model"
LOG_FILE="_resilient_${PREFIX}.log"
MAX_RETRIES="${RESILIENT_MAX_RETRIES:-20}"

latest_checkpoint_episode() {
    # Only numeric checkpoints count; best_0.pt is not a progress marker.
    ls "$MODEL_DIR" 2>/dev/null \
        | grep -E '^[0-9]+_0\.pt$' \
        | sed -E 's/^([0-9]+)_0\.pt$/\1/' \
        | sort -n | tail -1
}

attempt=0
while [ "$attempt" -lt "$MAX_RETRIES" ]; do
    attempt=$((attempt + 1))
    resume_ep="$(latest_checkpoint_episode)"

    if [ -n "$resume_ep" ] && [ "$resume_ep" -ge "$EPISODES" ]; then
        echo "[$(date)] $PREFIX already finished at episode $resume_ep." | tee -a "$LOG_FILE"
        exit 0
    fi

    if [ -n "$resume_ep" ] && [ "$resume_ep" -gt 0 ]; then
        echo "[$(date)] attempt $attempt: resuming $PREFIX from episode $resume_ep" | tee -a "$LOG_FILE"
        RESUME_ARGS=(--resume_episode "$resume_ep")
    else
        echo "[$(date)] attempt $attempt: starting $PREFIX from scratch" | tee -a "$LOG_FILE"
        RESUME_ARGS=()
    fi

    python3 run.py --task tsc --agent "$AGENT" --world "$WORLD" \
        --network "$NETWORK" --prefix "$PREFIX" --seed "$SEED" \
        "${RESUME_ARGS[@]}" "${EXTRA_ARGS[@]}" >> "$LOG_FILE" 2>&1

    exit_code=$?
    if [ "$exit_code" -eq 0 ]; then
        final_ep="$(latest_checkpoint_episode)"
        if [ -n "$final_ep" ] && [ "$final_ep" -ge "$EPISODES" ]; then
            echo "[$(date)] $PREFIX finished normally at episode $final_ep." | tee -a "$LOG_FILE"
            exit 0
        fi
        echo "[$(date)] $PREFIX exited 0 but checkpoints only reach ${final_ep:-none} (< $EPISODES); treating as an interruption." | tee -a "$LOG_FILE"
    else
        echo "[$(date)] $PREFIX exited $exit_code; retrying in 10s." | tee -a "$LOG_FILE"
    fi
    sleep 10
done

echo "[$(date)] $PREFIX hit the retry limit ($MAX_RETRIES); giving up. Check $LOG_FILE" | tee -a "$LOG_FILE"
exit 1
