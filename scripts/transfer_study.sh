#!/usr/bin/env bash
# Cross-network transfer study runner (container-side).
#
# Same working-directory / resilient-run machinery as scripts/chunk_study.sh:
# each job gets its own directory of symlinks back into the repo with configs/
# as a real copy, because run.py rewrites configs/sim/<network>.cfg with the
# run's seed and replay path at startup and concurrent jobs would race on it.
#
# Stages (see transfer/TRANSFER.md section 8):
#   stage1    4x4 source training: structural conditioning vs the learned-index
#             control, seeds 0/1/2, 250 episodes.  Produces the checkpoints that
#             every later stage transfers FROM.
#   zeroshot  evaluate the stage1 checkpoints on 16x3 / 7x28 without training.
#   finetune  50-episode fine-tune of the stage1 checkpoints on 16x3, plus the
#             from-scratch control.
#   compress  the two chunked cells of a {flat, chunked} x {structural, learned}
#             square; stage1's own runs are the two flat cells.
#   dynamic   structural conditioning with and without the slow traffic-state
#             term (dynamic/DYNAMIC.md).
#
# Usage (inside the container):
#   scripts/transfer_study.sh list                # print the job list and exit
#   scripts/transfer_study.sh stage1              # the 6 source runs
#   scripts/transfer_study.sh smoke               # 1 episode per config
#   scripts/transfer_study.sh zeroshot            # needs stage1 checkpoints
#   scripts/transfer_study.sh finetune            # needs stage1 checkpoints
#   scripts/transfer_study.sh compress            # chunked vs flat, same network
#   scripts/transfer_study.sh dynamic             # traffic-state conditioning
#
# Extra seeds for an existing stage are just a SEEDS override, e.g.
#   WORLD=sumo NETWORK=sumo1x21 SEEDS="3 4" scripts/transfer_study.sh stage1
#
# Environment overrides:
#   PARALLEL=6 SEEDS="0 1 2" EPISODES=250 NETWORK=cityflow4x4 WORLD=cityflow
#   TARGETS="cityflow16x3 cityflow7x28" FT_EPISODES=50
#
# PARALLEL: one job uses ~2.5 cores, so keep PARALLEL <= cores / 2.5.

set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="${1:-list}"

AGENT="${AGENT:-hyperlight_mappo}"
PARALLEL="${PARALLEL:-3}"
# cityflow | sumo. Ingolstadt21 (the only heterogeneous network available) lives
# in the SUMO world: WORLD=sumo NETWORK=sumo1x21.
WORLD="${WORLD:-cityflow}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export SIM_THREADS="${SIM_THREADS:-2}"
SAVE_RATE="${SAVE_RATE:-25}"
WORKROOT="${WORKROOT:-$REPO/tmp/transfer_study}"
STAMP_DIR="$WORKROOT/_status"
OUTROOT="${OUTROOT:-$REPO/data/output_data/tsc/${WORLD}_${AGENT}}"

# Each entry: <tag>|<extra run.py args>
#
# struct  : the thing under test.  meta comes only from network-independent
#           structural features, so the whole meta path can be transferred.
# learned : the control.  Per-intersection index table; on a different network
#           its embeddings are skipped by the shape filter and the hypernetwork
#           is fed randomly initialised codes.  It should lose, and if it does
#           not, the hypernetwork is not really using its conditioning.
CONFIGS=(
    "struct|--agent_embedding_mode structural"
    "learned|--agent_embedding_mode learned"
)

case "$STAGE" in
    stage1|list)
        NETWORK="${NETWORK:-cityflow4x4}"
        SEEDS="${SEEDS:-0 1 2}"
        EPISODES="${EPISODES:-250}"
        ;;
    smoke)
        NETWORK="${NETWORK:-cityflow4x4}"
        SEEDS="${SEEDS:-0}"
        EPISODES="${EPISODES:-1}"
        PARALLEL="${PARALLEL:-1}"
        SAVE_RATE="${SAVE_RATE:-1}"
        ;;
    compress)
        # Completes a {flat, chunked} x {structural, learned} square.  The two
        # flat cells are stage1's own runs -- stage1's `struct` is flat+
        # structural and `learned` is flat+learned (hyper_head_mode defaults to
        # flat) -- so only the two chunked cells are new here.
        NETWORK="${NETWORK:-cityflow4x4}"
        SEEDS="${SEEDS:-0 1 2}"
        EPISODES="${EPISODES:-250}"
        CONFIGS=(
            "c8struct|--agent_embedding_mode structural --hyper_head_mode chunked --hyper_chunk_size 8 --hyper_chunk_embed_dim 16"
            "c8learned|--agent_embedding_mode learned --hyper_head_mode chunked --hyper_chunk_size 8 --hyper_chunk_embed_dim 16"
        )
        ;;
    dynamic)
        # Isolates one variable: both arms condition on the structural features,
        # only one of them also gets the slow traffic-state term.  See
        # dynamic/DYNAMIC.md.  If the `struct` arm already exists on this
        # machine, resilient_run_world.sh sees its finished checkpoints and
        # returns immediately, so the control is never re-run needlessly.
        NETWORK="${NETWORK:-cityflow4x4}"
        SEEDS="${SEEDS:-0 1 2}"
        EPISODES="${EPISODES:-250}"
        CONFIGS=(
            "struct|--agent_embedding_mode structural"
            "structdyn|--agent_embedding_mode structural --dynamic_condition_enabled True"
        )
        ;;
    obsnorm)
        # Ports BRSC-MAPPO's incoming_occupancy normalisation: divide each
        # per-lane count by that lane's own storage rather than by a global
        # constant.  Ablating that dimension cost BRSC +29.3s of transferred
        # travel time (t=7.15), and the networks here differ enough in road
        # length (4x4 600-800m, 16x3 100-350m, 7x28 300m) for the fixed
        # constant to mean different things in each.  See transfer/observation.py.
        NETWORK="${NETWORK:-cityflow16x3}"
        SEEDS="${SEEDS:-0 1 2}"
        EPISODES="${EPISODES:-250}"
        CONFIGS=(
            "struct|--agent_embedding_mode structural"
            "structcap|--agent_embedding_mode structural --obs_norm_mode capacity"
        )
        ;;
    anneal)
        # Does decaying the learning rate (and the entropy bonus) settle the
        # oscillation that makes `last` a noisy sample on Ingolstadt? The
        # control is stage1's own `struct`, so only the two scheduled arms are
        # new here. Judge these on the spread of the late TEST points, not on
        # the final value.
        NETWORK="${NETWORK:-sumo1x21}"
        SEEDS="${SEEDS:-0 1 2}"
        EPISODES="${EPISODES:-250}"
        CONFIGS=(
            "structlr|--agent_embedding_mode structural --lr_anneal linear"
            "structlrent|--agent_embedding_mode structural --lr_anneal linear --entropy_anneal linear"
        )
        ;;
    baseline)
        # One non-HyperLight method per invocation, driven by AGENT. Output
        # already lands under <world>_<agent>/, so the tag can be the agent name
        # without any risk of collision:
        #   AGENT=dqn WORLD=sumo NETWORK=sumo1x21 scripts/transfer_study.sh baseline
        # mplight and frap are NOT runnable on sumo1x21: they look the network up
        # in a hand-written per-network block (phase_pairs / valid_acts /
        # lane_order) that exists for the grids, cologne1 and cologne3 only.
        NETWORK="${NETWORK:-cityflow4x4}"
        SEEDS="${SEEDS:-0 1 2}"
        EPISODES="${EPISODES:-250}"
        CONFIGS=("${AGENT}|")
        ;;
    zeroshot|finetune)
        NETWORK="${NETWORK:-cityflow4x4}"   # the SOURCE network of the checkpoints
        SEEDS="${SEEDS:-0 1 2}"
        TARGETS="${TARGETS:-cityflow16x3}"
        FT_EPISODES="${FT_EPISODES:-50}"
        ;;
    job)
        : # dispatched below
        ;;
    *)
        echo "unknown stage: $STAGE" >&2
        exit 2
        ;;
esac

source_checkpoint() {
    # stage1 saves the best-TEST checkpoint as model/best_0.pt
    local tag="$1" seed="$2"
    printf '%s' "$OUTROOT/$NETWORK/${tag}_${NETWORK#cityflow}_seed${seed}/model/best_0.pt"
}

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
    local network="$1" episodes="$2" prefix="$3" seed="$4" extra="$5"
    local dir
    dir="$(prepare_workdir "$prefix")"
    mkdir -p "$STAMP_DIR"

    echo "[$(date '+%F %T')] start $prefix  ($extra)"
    local status
    case "$extra" in
        *"--train_model False"*)
            # Evaluation-only job: no training means no checkpoint is ever
            # written, and resilient_run.sh judges success by "latest checkpoint
            # episode >= EPISODES".  It would therefore call every successful
            # evaluation an interruption and re-run it MAX_RETRIES times, so
            # eval-only jobs call run.py directly instead.
            ( cd "$dir" && python3 run.py --task tsc --agent "$AGENT" --world "$WORLD" \
                --network "$network" --prefix "$prefix" --seed "$seed" \
                --thread_num "$SIM_THREADS" $extra >> "_eval_${prefix}.log" 2>&1 )
            status=$?
            ;;
        *)
            # resilient_run_world.sh rather than resilient_run.sh: the latter
            # hardcodes --world cityflow, and it is mid-run on the long 7x28
            # jobs, where editing it in place would corrupt a live run.
            RESILIENT_WORLD="$WORLD" RESILIENT_NETWORK="$network" \
                RESILIENT_AGENT="$AGENT" RESILIENT_EPISODES="$episodes" \
                "$dir/scripts/resilient_run_world.sh" "$prefix" "$seed" \
                --episodes "$episodes" --save_rate "$SAVE_RATE" --thread_num "$SIM_THREADS" $extra
            status=$?
            ;;
    esac
    echo "$status" > "$STAMP_DIR/$prefix.exit"
    echo "[$(date '+%F %T')] done  $prefix (exit=$status)"
    return $status
}

if [ "$STAGE" = "job" ]; then
    # scripts/transfer_study.sh job <network> <episodes> <prefix> <seed> <extra args...>
    shift
    network="$1"; episodes="$2"; prefix="$3"; seed="$4"; shift 4
    run_job "$network" "$episodes" "$prefix" "$seed" "$*"
    exit $?
fi

JOBS=()
case "$STAGE" in
    stage1|list|smoke|compress|dynamic|obsnorm|baseline|anneal)
        # smoke gets its own prefix so a 1-episode validation run can never be
        # mistaken for -- or resumed as -- a real stage1 run.
        name_prefix=''
        [ "$STAGE" = "smoke" ] && name_prefix='smoke_'
        for entry in "${CONFIGS[@]}"; do
            tag="${entry%%|*}"
            extra="${entry#*|}"
            for seed in $SEEDS; do
                prefix="${name_prefix}${tag}_${NETWORK#cityflow}_seed${seed}"
                JOBS+=("$NETWORK|$EPISODES|$prefix|$seed|$extra")
            done
        done
        ;;
    zeroshot)
        # --train_model False is what makes this a *zero-shot* evaluation: with
        # load_model also false, task.py calls trainer.test(drop_load=True), so
        # the agent is evaluated exactly as the transfer checkpoint left it,
        # with no gradient step taken on the target network.
        for entry in "${CONFIGS[@]}"; do
            tag="${entry%%|*}"
            extra="${entry#*|}"
            for seed in $SEEDS; do
                ckpt="$(source_checkpoint "$tag" "$seed")"
                for target in $TARGETS; do
                    prefix="zs_${tag}_${NETWORK#cityflow}to${target#cityflow}_seed${seed}"
                    JOBS+=("$target|1|$prefix|$seed|$extra --train_model False --transfer_checkpoint $ckpt")
                done
            done
        done
        ;;
    finetune)
        for entry in "${CONFIGS[@]}"; do
            tag="${entry%%|*}"
            extra="${entry#*|}"
            for seed in $SEEDS; do
                ckpt="$(source_checkpoint "$tag" "$seed")"
                for target in $TARGETS; do
                    prefix="ft_${tag}_${NETWORK#cityflow}to${target#cityflow}_seed${seed}"
                    JOBS+=("$target|$FT_EPISODES|$prefix|$seed|$extra --transfer_checkpoint $ckpt")
                done
            done
        done
        # from-scratch control: same budget, no transfer
        for seed in $SEEDS; do
            for target in $TARGETS; do
                prefix="ft_scratch_${target#cityflow}_seed${seed}"
                JOBS+=("$target|$FT_EPISODES|$prefix|$seed|--agent_embedding_mode structural")
            done
        done
        ;;
esac

echo "stage=$STAGE parallel=$PARALLEL jobs=${#JOBS[@]}"
for job in "${JOBS[@]}"; do
    IFS='|' read -r n e p s x <<< "$job"
    echo "  $p  [$n, ${e}ep]  $x"
done

if [ "$STAGE" = "list" ]; then
    exit 0
fi

# Fail early rather than after hours: a missing source checkpoint means stage1
# has not finished (or used a different prefix).
if [ "$STAGE" = "zeroshot" ] || [ "$STAGE" = "finetune" ]; then
    missing=0
    for job in "${JOBS[@]}"; do
        ckpt="${job##*--transfer_checkpoint }"
        case "$job" in
            *--transfer_checkpoint*)
                if [ ! -f "$ckpt" ]; then
                    echo "missing source checkpoint: $ckpt" >&2
                    missing=$((missing + 1))
                fi
                ;;
        esac
    done
    if [ "$missing" -gt 0 ]; then
        echo "refusing to start: $missing source checkpoint(s) missing; run stage1 first" >&2
        exit 3
    fi
fi

mkdir -p "$WORKROOT" "$STAMP_DIR"
running=0
for job in "${JOBS[@]}"; do
    IFS='|' read -r n e p s x <<< "$job"
    "$REPO/scripts/transfer_study.sh" job "$n" "$e" "$p" "$s" $x &
    running=$((running + 1))
    if [ "$running" -ge "$PARALLEL" ]; then
        wait -n
        running=$((running - 1))
    fi
    # Stagger starts: CityFlow parses the roadnet single-threaded.
    sleep 20
done
wait

echo "[$(date '+%F %T')] all jobs finished"
for f in "$STAMP_DIR"/*.exit; do
    [ -e "$f" ] || continue
    echo "  $(basename "$f" .exit): exit=$(cat "$f")"
done
