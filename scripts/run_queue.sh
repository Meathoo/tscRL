#!/usr/bin/env bash
# Load-aware job queue for the 7x28 stage of the chunked study.
#
# scripts/chunk_study.sh's own pool assumes it owns the machine. Here jobs from
# an earlier launch are still running, and re-invoking a live run would start a
# second process resuming from the same checkpoint directory. So this queue
# counts live run.py processes instead of tracking its own children, and only
# dispatches while the machine is below TARGET_JOBS.
#
# Usage (inside the container):
#   scripts/run_queue.sh queue.txt          # one "<tag> <seed> <extra args...>" per line
#   TARGET_JOBS=4 NETWORK=cityflow7x28 scripts/run_queue.sh queue.txt

set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QUEUE_FILE="${1:?usage: run_queue.sh <queue file>}"
NETWORK="${NETWORK:-cityflow7x28}"
EPISODES="${EPISODES:-250}"
TARGET_JOBS="${TARGET_JOBS:-4}"
POLL_SECONDS="${POLL_SECONDS:-300}"

live_jobs() {
    # pgrep -c prints 0 and exits 1 when nothing matches, so a `|| echo 0`
    # fallback would emit the count twice and break the numeric comparison.
    local count
    count=$(pgrep -fc '[r]un\.py --task tsc' 2>/dev/null)
    echo "${count:-0}"
}

mkdir -p "$REPO/tmp"

while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in ''|'#'*) continue ;; esac
    read -r tag seed extra <<< "$line"

    while [ "$(live_jobs)" -ge "$TARGET_JOBS" ]; do
        sleep "$POLL_SECONDS"
    done

    echo "[$(date '+%F %T')] dispatch ${tag}_seed${seed} (live=$(live_jobs))"
    setsid bash "$REPO/scripts/chunk_study.sh" job "$NETWORK" "$EPISODES" "$tag" "$seed" $extra \
        >> "$REPO/tmp/queue_${tag}_seed${seed}.log" 2>&1 &
    # Give the new process time to appear in the live count before re-checking,
    # and stagger CityFlow's single-threaded roadnet load across slots.
    sleep 60
done < "$QUEUE_FILE"

echo "[$(date '+%F %T')] queue drained; waiting for stragglers"
wait
echo "[$(date '+%F %T')] all queued jobs finished"
