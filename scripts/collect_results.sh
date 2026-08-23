#!/usr/bin/env bash
# Pull the DTL/BRF logs of every run off the lab boxes into results/.
#
# Runs live on three machines and some prefixes collide across them -- .232 has
# the static Ingolstadt 2x2 as struct_sumo1x21_seed0..4 while .237 has the
# dynamic study's control arm under the *same* names -- so the tree is
# namespaced by machine rather than merged. Nothing here writes into
# data/output_data.
#
# Only logs are copied (a few KB each); checkpoints and replays stay put.
#
# Usage (on the workstation, not in the container):
#   scripts/collect_results.sh            # all machines
#   scripts/collect_results.sh local      # one machine

set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${DEST:-$REPO/results}"
KEY="${KEY:-$HOME/.ssh/id_ed25519_nsysu}"
WHICH="${1:-all}"

# name|ssh target|remote repo root ('-' means this machine)
MACHINES=(
    "local|-|$REPO"
    "m232|m143040017@140.117.172.232|/home/m143040017/tscRL_transfer"
    "m237|m143040017@140.117.172.237|/home/m143040017/tscRL"
)

# tar over ssh rather than rsync: the Windows workstation this is driven from
# has no rsync, and the payload is a few thousand small text files.
# A single -regex keeps the pattern inside quotes, so no parentheses have to
# survive an extra round of shell parsing on the remote side.
FIND_EXPR="-type f -regex '.*\\(DTL\\.log\\|BRF\\.log\\|hyperparameters\\.json\\)'"

copy_local() {
    local root="$1" dest="$2"
    local src="$root/data/output_data/tsc"
    [ -d "$src" ] || { echo "  no data/output_data/tsc under $root"; return; }
    ( cd "$src" && eval "find . $FIND_EXPR -print0" \
        | tar czf - --null -T - ) | tar xzf - -C "$dest"
}

copy_remote() {
    local target="$1" root="$2" dest="$3"
    ssh -o BatchMode=yes -i "$KEY" "$target" \
        "cd $root/data/output_data/tsc 2>/dev/null && find . $FIND_EXPR -print0 | tar czf - --null -T -" \
        | tar xzf - -C "$dest"
}

for entry in "${MACHINES[@]}"; do
    IFS='|' read -r name target root <<< "$entry"
    if [ "$WHICH" != "all" ] && [ "$WHICH" != "$name" ]; then
        continue
    fi
    dest="$DEST/$name"
    mkdir -p "$dest"
    echo "[$(date '+%F %T')] collecting $name -> $dest"
    if [ "$target" = '-' ]; then
        copy_local "$root" "$dest"
    else
        copy_remote "$target" "$root" "$dest"
    fi
    count=$(find "$dest" -name '*DTL.log' | wc -l)
    echo "  $count DTL logs"
done

echo "[$(date '+%F %T')] done. Summarize with e.g.:"
echo "  python scripts/summarize_chunk_study.py --root $DEST/m232 --world sumo --network sumo1x21"
