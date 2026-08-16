#!/usr/bin/env bash
# Chunked-hypernetwork study runner (container-side).
#
# Runs a matrix of (config, seed) jobs with a fixed number of parallel slots.
# Each job gets its own working directory of symlinks back into the repo, with
# configs/ as a real copy: run.py rewrites configs/sim/<network>.cfg with the
# run's seed and replay path at startup, so jobs sharing one working directory
# would race on that file. data/ stays symlinked, so all runs still write into
# the single data/output_data tree.
#
# Usage (inside the container):
#   scripts/chunk_study.sh stage1            # 4x4 screening matrix
#   scripts/chunk_study.sh stage2            # 7x28 confirmation matrix
#   scripts/chunk_study.sh smoke             # 1 episode per config, for validation
#   scripts/chunk_study.sh list              # print the job list and exit
#
# Environment overrides:
#   PARALLEL=4 SEEDS="0 1 2" EPISODES=250 NETWORK=cityflow4x4 AGENT=hyperlight_mappo

set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="${1:-list}"

AGENT="${AGENT:-hyperlight_mappo}"
PARALLEL="${PARALLEL:-2}"

# The box has 6 physical cores. Left alone, every job spawns ~32 threads (torch
# defaults to one per physical core, plus CityFlow's own pool), so a handful of
# concurrent runs oversubscribe the machine several times over and each one
# crawls. The actor/critic MLPs here are tiny and gain nothing from wide intra-op
# parallelism, so cap the math threads and keep the sim pool small.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export SIM_THREADS="${SIM_THREADS:-2}"
SAVE_RATE="${SAVE_RATE:-25}"
WORKROOT="${WORKROOT:-$REPO/tmp/chunk_study}"
STAMP_DIR="$WORKROOT/_status"

# Each entry: <tag>|<extra run.py args>
# aw / c8 are the two references from docs/HYPERNETWORK_COMPRESSION_METHODS.md
# and must be re-run here: this machine has no data/output_data from the
# earlier experiments, so nothing is comparable across machines.
CONFIGS=(
    "aw|--hyper_head_mode flat"
    "c8|--hyper_head_mode chunked --hyper_chunk_size 8 --hyper_chunk_embed_dim 16"
    "c8rf|--hyper_head_mode chunked --hyper_chunk_size 8 --hyper_chunk_embed_dim 16 --hyper_rf_init True"
    "c8g64|--hyper_head_mode chunked --hyper_chunk_size 8 --hyper_chunk_embed_dim 16 --hyper_chunk_generator_hidden 64"
    "c8g64hh256|--hyper_head_mode chunked --hyper_chunk_size 8 --hyper_chunk_embed_dim 16 --hyper_chunk_generator_hidden 64 --hyper_hidden 256"
    "c8split|--hyper_head_mode chunked --hyper_chunk_embed_dim 16 --hyper_actor_chunk_size 16 --hyper_critic_chunk_size 4"
    "c8res|--hyper_head_mode chunked --hyper_chunk_size 8 --hyper_chunk_embed_dim 16 --hyper_residual True --hyper_residual_mode full --hyper_residual_scale 1.0"
)

case "$STAGE" in
    stage1|list)
        NETWORK="${NETWORK:-cityflow4x4}"
        SEEDS="${SEEDS:-0 1 2}"
        EPISODES="${EPISODES:-250}"
        ;;
    stage2)
        # Narrowed by stage 1; STAGE2_TAGS keeps the winners plus both references.
        NETWORK="${NETWORK:-cityflow7x28}"
        SEEDS="${SEEDS:-0 1 2}"
        EPISODES="${EPISODES:-250}"
        STAGE2_TAGS="${STAGE2_TAGS:-aw c8 c8g64hh256}"
        filtered=()
        for entry in "${CONFIGS[@]}"; do
            tag="${entry%%|*}"
            for keep in $STAGE2_TAGS; do
                if [ "$tag" = "$keep" ]; then filtered+=("$entry"); fi
            done
        done
        CONFIGS=("${filtered[@]}")
        ;;
    smoke)
        NETWORK="${NETWORK:-cityflow4x4}"
        SEEDS="${SEEDS:-0}"
        EPISODES="${EPISODES:-1}"
        PARALLEL="${PARALLEL:-1}"
        ;;
    job)
        : # dispatched below
        ;;
    *)
        echo "unknown stage: $STAGE" >&2
        exit 2
        ;;
esac

prepare_workdir() {
    local prefix="$1"
    local dir="$WORKROOT/$prefix"
    mkdir -p "$dir"
    for entry in "$REPO"/*; do
        local base
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

run_job() {
    local network="$1" episodes="$2" tag="$3" seed="$4" extra="$5"
    local prefix="${tag}_${network#cityflow}_seed${seed}"
    local dir
    dir="$(prepare_workdir "$prefix")"
    mkdir -p "$STAMP_DIR"

    echo "[$(date '+%F %T')] start $prefix  ($extra)"
    RESILIENT_NETWORK="$network" RESILIENT_AGENT="$AGENT" RESILIENT_EPISODES="$episodes" \
        "$dir/resilient_run.sh" "$prefix" "$seed" \
        --episodes "$episodes" --save_rate "$SAVE_RATE" --thread_num "$SIM_THREADS" $extra
    local status=$?
    echo "$status" > "$STAMP_DIR/$prefix.exit"
    echo "[$(date '+%F %T')] done  $prefix (exit=$status)"
    return $status
}

if [ "$STAGE" = "job" ]; then
    # scripts/chunk_study.sh job <network> <episodes> <tag> <seed> <extra args...>
    shift
    network="$1"; episodes="$2"; tag="$3"; seed="$4"; shift 4
    run_job "$network" "$episodes" "$tag" "$seed" "$*"
    exit $?
fi

JOBS=()
for entry in "${CONFIGS[@]}"; do
    tag="${entry%%|*}"
    extra="${entry#*|}"
    for seed in $SEEDS; do
        JOBS+=("$NETWORK|$EPISODES|$tag|$seed|$extra")
    done
done

echo "stage=$STAGE network=$NETWORK episodes=$EPISODES seeds=[$SEEDS] parallel=$PARALLEL jobs=${#JOBS[@]}"
for job in "${JOBS[@]}"; do
    IFS='|' read -r n e t s x <<< "$job"
    echo "  ${t}_${n#cityflow}_seed${s}  $x"
done

if [ "$STAGE" = "list" ]; then
    exit 0
fi

mkdir -p "$WORKROOT" "$STAMP_DIR"
running=0
for job in "${JOBS[@]}"; do
    IFS='|' read -r n e t s x <<< "$job"
    "$REPO/scripts/chunk_study.sh" job "$n" "$e" "$t" "$s" $x &
    running=$((running + 1))
    if [ "$running" -ge "$PARALLEL" ]; then
        wait -n
        running=$((running - 1))
    fi
    # Stagger starts: CityFlow loads the roadnet single-threaded, and spacing the
    # launches keeps the parallel slots from all doing that at the same time.
    sleep 20
done
wait

echo "[$(date '+%F %T')] all jobs finished"
grep -c . "$STAMP_DIR"/*.exit >/dev/null 2>&1 || true
for f in "$STAMP_DIR"/*.exit; do
    [ -e "$f" ] || continue
    echo "  $(basename "$f" .exit): exit=$(cat "$f")"
done
