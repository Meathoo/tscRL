#!/usr/bin/env bash
# Ingolstadt21 (SUMO) study runner (container-side).
#
# Same variant matrix as scripts/chunk_study.sh, but on the SUMO world instead
# of CityFlow: --world sumo --network sumo1x21, which configs/sim/sumo1x21.cfg
# points at data/raw_data/ingolstadt21. That map has 21 signals with unequal
# phase counts (action_dim = 4 = the max), so it is the heterogeneous-network
# check the CityFlow grids cannot give.
#
# This is a separate file rather than a flag on chunk_study.sh/resilient_run.sh
# on purpose: those two are executing right now for the 7x28 jobs, and bash
# re-reads a running script from a byte offset, so editing them in place can
# corrupt a live run.
#
# Usage (inside the container):
#   scripts/ingolstadt_study.sh list                 # print the job list
#   scripts/ingolstadt_study.sh job <tag> <seed>     # one job, with resume-on-crash
#   scripts/ingolstadt_study.sh run                  # load-aware dispatch of TAGS x SEEDS
#
# Environment overrides:
#   TAGS="aw c8rf" SEEDS="0 1 2" EPISODES=250 TARGET_JOBS=4 NETWORK=sumo1x21

set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-list}"

WORLD="${WORLD:-sumo}"
NETWORK="${NETWORK:-sumo1x21}"
AGENT="${AGENT:-hyperlight_mappo}"
EPISODES="${EPISODES:-250}"
SAVE_RATE="${SAVE_RATE:-25}"
SEEDS="${SEEDS:-0}"
TAGS="${TAGS:-aw c8 c8rf c8g64rf}"
# Total live run.py processes tolerated on the box: this study's jobs plus any
# CityFlow job still running from the chunked study.
TARGET_JOBS="${TARGET_JOBS:-4}"
POLL_SECONDS="${POLL_SECONDS:-300}"
MAX_RETRIES="${MAX_RETRIES:-20}"

# libsumo is single-threaded, so a job is one sim thread plus torch's pool.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

WORKROOT="${WORKROOT:-$REPO/tmp/ingolstadt}"
STAMP_DIR="$WORKROOT/_status"

# Tag -> extra run.py args. Kept identical to chunk_study.sh so the Ingolstadt
# numbers line up with the 4x4 / 7x28 tables in
# docs/HYPERNETWORK_COMPRESSION_METHODS.md.
config_args() {
    case "$1" in
        aw)          echo "--hyper_head_mode flat" ;;
        c8)          echo "--hyper_head_mode chunked --hyper_chunk_size 8 --hyper_chunk_embed_dim 16" ;;
        c8rf)        echo "--hyper_head_mode chunked --hyper_chunk_size 8 --hyper_chunk_embed_dim 16 --hyper_rf_init True" ;;
        c8g64)       echo "--hyper_head_mode chunked --hyper_chunk_size 8 --hyper_chunk_embed_dim 16 --hyper_chunk_generator_hidden 64" ;;
        c8g64rf)     echo "--hyper_head_mode chunked --hyper_chunk_size 8 --hyper_chunk_embed_dim 16 --hyper_chunk_generator_hidden 64 --hyper_rf_init True" ;;
        # per_chunk rf init: byte-for-byte the same run as c8rf / c8g64rf apart
        # from where rf_init writes the target-layer init, and the same parameter
        # count. Compare seed spread, not just the mean.
        c8rfpc)      echo "--hyper_head_mode chunked --hyper_chunk_size 8 --hyper_chunk_embed_dim 16 --hyper_rf_init True --hyper_chunk_rf_mode per_chunk" ;;
        c8g64rfpc)   echo "--hyper_head_mode chunked --hyper_chunk_size 8 --hyper_chunk_embed_dim 16 --hyper_chunk_generator_hidden 64 --hyper_rf_init True --hyper_chunk_rf_mode per_chunk" ;;
        c8g64hh256)  echo "--hyper_head_mode chunked --hyper_chunk_size 8 --hyper_chunk_embed_dim 16 --hyper_chunk_generator_hidden 64 --hyper_hidden 256" ;;
        c8split)     echo "--hyper_head_mode chunked --hyper_chunk_embed_dim 16 --hyper_actor_chunk_size 16 --hyper_critic_chunk_size 4" ;;
        c8res)       echo "--hyper_head_mode chunked --hyper_chunk_size 8 --hyper_chunk_embed_dim 16 --hyper_residual True --hyper_residual_mode full --hyper_residual_scale 1.0" ;;
        *)           echo "__UNKNOWN__" ;;
    esac
}

# run.py rewrites configs/sim/<network>.cfg with the run's seed and replay path
# at startup, so each job needs its own configs/ copy. Everything else is
# symlinked back into the repo, data/ included, so all runs share one
# data/output_data tree.
prepare_workdir() {
    local prefix="$1"
    local dir="$WORKROOT/$prefix"
    mkdir -p "$dir"
    local entry base
    for entry in "$REPO"/*; do
        base="$(basename "$entry")"
        case "$base" in
            configs|tmp) continue ;;
        esac
        ln -sfn "$entry" "$dir/$base"
    done
    rm -rf "$dir/configs"
    cp -r "$REPO/configs" "$dir/configs"
    printf '%s' "$dir"
}

live_jobs() {
    local count
    count=$(pgrep -fc '[r]un\.py --task tsc' 2>/dev/null)
    echo "${count:-0}"
}

latest_checkpoint() {
    ls "$1" 2>/dev/null | grep -E '^[0-9]+_0\.pt$' | sed -E 's/^([0-9]+)_0\.pt$/\1/' | sort -n | tail -1
}

# One job, restarted from its newest checkpoint if it dies (the WSL/Docker
# interruptions resilient_run.sh was written for).
run_job() {
    local tag="$1" seed="$2"
    local extra
    extra="$(config_args "$tag")"
    if [ "$extra" = "__UNKNOWN__" ]; then
        echo "unknown tag: $tag" >&2
        return 2
    fi

    local prefix="${tag}_ing21_seed${seed}"
    local dir
    dir="$(prepare_workdir "$prefix")"
    local model_dir="$dir/data/output_data/tsc/${WORLD}_${AGENT}/${NETWORK}/${prefix}/model"
    local log_file="$dir/_ing21_${prefix}.log"
    mkdir -p "$STAMP_DIR"

    local attempt=0 resume_ep exit_code final_ep
    while [ "$attempt" -lt "$MAX_RETRIES" ]; do
        attempt=$((attempt + 1))
        resume_ep="$(latest_checkpoint "$model_dir")"

        if [ -n "$resume_ep" ] && [ "$resume_ep" -ge "$EPISODES" ]; then
            echo "[$(date '+%F %T')] $prefix already at episode $resume_ep; done." | tee -a "$log_file"
            echo 0 > "$STAMP_DIR/$prefix.exit"
            return 0
        fi

        local resume_args=()
        if [ -n "$resume_ep" ] && [ "$resume_ep" -gt 0 ]; then
            echo "[$(date '+%F %T')] attempt $attempt: resuming $prefix from episode $resume_ep" | tee -a "$log_file"
            resume_args=(--resume_episode "$resume_ep")
        else
            echo "[$(date '+%F %T')] attempt $attempt: starting $prefix from scratch ($extra)" | tee -a "$log_file"
        fi

        ( cd "$dir" && python3 run.py --task tsc --agent "$AGENT" --world "$WORLD" \
            --network "$NETWORK" --prefix "$prefix" --seed "$seed" \
            --episodes "$EPISODES" --save_rate "$SAVE_RATE" \
            "${resume_args[@]}" $extra >> "$log_file" 2>&1 )
        exit_code=$?

        if [ "$exit_code" -eq 0 ]; then
            final_ep="$(latest_checkpoint "$model_dir")"
            if [ -n "$final_ep" ] && [ "$final_ep" -ge "$EPISODES" ]; then
                echo "[$(date '+%F %T')] $prefix finished at episode $final_ep." | tee -a "$log_file"
                echo 0 > "$STAMP_DIR/$prefix.exit"
                return 0
            fi
            echo "[$(date '+%F %T')] $prefix exited 0 but only reached ${final_ep:-none}/$EPISODES; retrying." | tee -a "$log_file"
        else
            echo "[$(date '+%F %T')] $prefix died (exit=$exit_code); retrying in 10s." | tee -a "$log_file"
        fi
        sleep 10
    done

    echo "[$(date '+%F %T')] $prefix gave up after $MAX_RETRIES attempts; see $log_file" | tee -a "$log_file"
    echo 1 > "$STAMP_DIR/$prefix.exit"
    return 1
}

case "$MODE" in
    list)
        for tag in $TAGS; do
            for seed in $SEEDS; do
                echo "${tag}_ing21_seed${seed}  ($(config_args "$tag"))"
            done
        done
        ;;
    job)
        run_job "${2:?usage: ingolstadt_study.sh job <tag> <seed>}" "${3:?usage: ingolstadt_study.sh job <tag> <seed>}"
        ;;
    run)
        mkdir -p "$WORKROOT"
        for tag in $TAGS; do
            for seed in $SEEDS; do
                while [ "$(live_jobs)" -ge "$TARGET_JOBS" ]; do
                    sleep "$POLL_SECONDS"
                done
                echo "[$(date '+%F %T')] dispatch ${tag}_ing21_seed${seed} (live=$(live_jobs))"
                setsid env EPISODES="$EPISODES" SAVE_RATE="$SAVE_RATE" MAX_RETRIES="$MAX_RETRIES" \
                    bash "$REPO/scripts/ingolstadt_study.sh" job "$tag" "$seed" \
                    >> "$WORKROOT/dispatch_${tag}_seed${seed}.log" 2>&1 &
                # Let the new process appear in the live count before re-checking.
                sleep 60
            done
        done
        echo "[$(date '+%F %T')] all jobs dispatched; waiting"
        wait
        echo "[$(date '+%F %T')] all jobs finished"
        ;;
    *)
        echo "unknown mode: $MODE" >&2
        exit 2
        ;;
esac
