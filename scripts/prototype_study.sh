#!/usr/bin/env bash
# K-prototype study runner (container-side).
#
# Sweeps K in the prototype-factorized head (docs/NEXT_ARCHITECTURE_PROPOSALS.md
# sec 6). The point of the sweep is that its two ends are already measured:
#
#   k0  is the existing `struct` arm -- hyper_prototypes 0 constructs nothing
#       and is bit-identical to it.
#   k1  is constmeta by construction (the generated parameters have exactly
#       zero spread across intersections), so it has to reproduce (q)'s
#       failure: worse than a plain shared MLP.
#
# If those two ends do not land where PROGRESS.md sec 6 (o-2)/(q) say they
# land, nothing in the middle is worth reading, and the first thing to suspect
# is this runner rather than the result.
#
# Ingolstadt21 is the network that can separate. On the CityFlow grids the
# structural contract is 10 constants out of 12, so the gate has almost nothing
# to read and the arm is predicted null there for the same reason F1/F2 were --
# tests/test_prototype_hypernetwork.py pins that as a property of the
# construction, not a hope. cityflow4x4_hetero is run anyway, because every
# null in this study so far is on CityFlow and every win is on SUMO, and that
# confound has to stop growing (PROGRESS.md sec 6 (t-3)).
#
# Standalone copy of the chunk_study/ingolstadt_study dispatch logic on
# purpose: bash re-reads a running script from a byte offset, so editing one
# that has live jobs attached can corrupt them.
#
# Usage (inside the container):
#   scripts/prototype_study.sh list                  # print the job list
#   scripts/prototype_study.sh job <tag> <seed>      # one job, resume-on-crash
#   scripts/prototype_study.sh run                   # load-aware dispatch
#
# Environment overrides:
#   TAGS="k0 k1 k8 k8f" SEEDS="0 1 2" EPISODES=250 TARGET_JOBS=6
#   WORLD=sumo NETWORK=sumo1x21

set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-list}"

WORLD="${WORLD:-sumo}"
NETWORK="${NETWORK:-sumo1x21}"
AGENT="${AGENT:-hyperlight_mappo}"
EPISODES="${EPISODES:-250}"
SAVE_RATE="${SAVE_RATE:-25}"
SEEDS="${SEEDS:-0 1 2}"
# Screening set. k2/k4/k16 fill in the curve once k8 has shown whether there is
# anything to fill in; running all seven tags up front is ~40h of this box.
TAGS="${TAGS:-k0 k1 k8 k8f}"
TARGET_JOBS="${TARGET_JOBS:-6}"
POLL_SECONDS="${POLL_SECONDS:-300}"
MAX_RETRIES="${MAX_RETRIES:-20}"

# One sim thread plus torch's pool. This must not vary across the sweep: the
# thread configuration is what breaks run-to-run comparability on a box, not
# the box (see the determinism note in PROGRESS.md sec 6.4).
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
THREAD_NUM="${THREAD_NUM:-2}"

WORKROOT="${WORKROOT:-$REPO/tmp/prototype}"
STAMP_DIR="$WORKROOT/_status"

# Every arm holds agent_embedding_mode fixed at structural and varies only K,
# so the sweep is a clean one-variable axis. The conditioning mode is the other
# axis and is deliberately not crossed with this one here.
BASE_ARGS="--agent_embedding_mode structural"

config_args() {
    case "$1" in
        k0)   echo "$BASE_ARGS --hyper_prototypes 0" ;;
        k1)   echo "$BASE_ARGS --hyper_prototypes 1" ;;
        k2)   echo "$BASE_ARGS --hyper_prototypes 2" ;;
        k4)   echo "$BASE_ARGS --hyper_prototypes 4" ;;
        k8)   echo "$BASE_ARGS --hyper_prototypes 8" ;;
        k16)  echo "$BASE_ARGS --hyper_prototypes 16" ;;
        # The C3 control: gate frozen at its random init, so the partition is
        # fixed and arbitrary. Separates "K sets of weights" (capacity) from
        # "structurally alike intersections should share a policy" (the claim).
        k8f)  echo "$BASE_ARGS --hyper_prototypes 8 --hyper_prototype_gate_frozen True" ;;
        # K=N on Ingolstadt21, for the far end of the axis. Not a substitute for
        # the `learned` arm: this keeps the structural meta and only widens K.
        k21)  echo "$BASE_ARGS --hyper_prototypes 21" ;;
        *)    echo "__UNKNOWN__" ;;
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

job_prefix() {
    echo "proto_$1_${NETWORK}_seed$2"
}

run_job() {
    local tag="$1" seed="$2"
    local extra
    extra="$(config_args "$tag")"
    if [ "$extra" = "__UNKNOWN__" ]; then
        echo "unknown tag: $tag" >&2
        return 2
    fi

    local prefix
    prefix="$(job_prefix "$tag" "$seed")"
    local dir
    dir="$(prepare_workdir "$prefix")"
    local model_dir="$dir/data/output_data/tsc/${WORLD}_${AGENT}/${NETWORK}/${prefix}/model"
    local log_file="$dir/_proto_${prefix}.log"
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
            --thread_num "$THREAD_NUM" \
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
                echo "$(job_prefix "$tag" "$seed")  ($(config_args "$tag"))"
            done
        done
        ;;
    job)
        run_job "${2:?usage: prototype_study.sh job <tag> <seed>}" \
                "${3:?usage: prototype_study.sh job <tag> <seed>}"
        ;;
    run)
        mkdir -p "$WORKROOT"
        for tag in $TAGS; do
            for seed in $SEEDS; do
                while [ "$(live_jobs)" -ge "$TARGET_JOBS" ]; do
                    sleep "$POLL_SECONDS"
                done
                echo "[$(date '+%F %T')] dispatching $(job_prefix "$tag" "$seed")"
                run_job "$tag" "$seed" &
                sleep 20
            done
        done
        wait
        echo "[$(date '+%F %T')] all jobs finished"
        ;;
    status)
        for tag in $TAGS; do
            for seed in $SEEDS; do
                prefix="$(job_prefix "$tag" "$seed")"
                model_dir="$WORKROOT/$prefix/data/output_data/tsc/${WORLD}_${AGENT}/${NETWORK}/${prefix}/model"
                echo "$prefix: episode $(latest_checkpoint "$model_dir" || echo none)/$EPISODES"
            done
        done
        ;;
    *)
        echo "usage: prototype_study.sh {list|job <tag> <seed>|run|status}" >&2
        exit 2
        ;;
esac
