"""Read prototype occupancy back out of finished checkpoints.

The K-prototype head keeps its gate occupancy in a registered buffer, so a
checkpoint carries it whether or not anything logged it during the run --
which matters here, because `HyperLightPPOAgent.current_schedule()` has no
caller anywhere in the repo, so neither these diagnostics nor the existing
lr/entropy anneal diagnostics ever reached a log file.

What to look for: `active` collapsing to 1 means the gate has concentrated all
of its mass on a single prototype, which makes theta identical across
intersections -- constmeta, which (q) measured losing to a plain shared MLP by
61.7s on Ingolstadt21. An arm that collapsed is not a K=8 result no matter what
K was passed on the command line.

Usage (inside the container):

    python scripts/prototype_diagnostics.py                    # every proto_* run
    python scripts/prototype_diagnostics.py --match proto_k8   # one arm
    python scripts/prototype_diagnostics.py --episode 250      # a fixed episode

The study runners give each job its own work dir of symlinks but share one
data/output_data tree, so runs are found there rather than under tmp/.
"""

import argparse
import glob
import math
import os
import re
import sys

import torch


def _entropy(share):
    return -sum(s * math.log(s) for s in share if s > 0)


def summarise_head(state, threshold=0.01):
    usage = state.get('usage')
    if usage is None:
        return None
    usage = usage.detach().float()
    total = float(usage.sum())
    if total <= 0:
        return None
    share = (usage / total).tolist()
    k = len(share)
    return {
        'k': k,
        'active': sum(1 for s in share if s > threshold),
        'entropy': _entropy(share),
        # Perplexity is the entropy read as "how many prototypes is this
        # equivalent to using evenly", which is the number worth comparing
        # against the K that was requested.
        'perplexity': math.exp(_entropy(share)),
        'max_share': max(share),
        'temperature': float(state['temperature']) if 'temperature' in state else float('nan'),
        'share': share,
    }


def checkpoint_for(run_dir, episode=None):
    paths = glob.glob(os.path.join(run_dir, 'model', '*_0.pt'))
    if not paths:
        return None
    if episode is not None:
        exact = [p for p in paths if os.path.basename(p) == f'{episode}_0.pt']
        if exact:
            return exact[0]
    numbered = []
    for path in paths:
        match = re.match(r'^(\d+)_0\.pt$', os.path.basename(path))
        if match:
            numbered.append((int(match.group(1)), path))
    if not numbered:
        return None
    return max(numbered)[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', default='data/output_data/tsc',
                        help='the output_data/tsc tree the runs write into')
    parser.add_argument('--match', default='proto_',
                        help='only runs whose directory name starts with this')
    parser.add_argument('--episode', type=int, default=None,
                        help='read this episode instead of the newest checkpoint')
    parser.add_argument('--threshold', type=float, default=0.01,
                        help='share above which a prototype counts as active')
    args = parser.parse_args()

    runs = sorted(
        d for d in glob.glob(os.path.join(args.root, '*', '*', '*'))
        if os.path.isdir(d) and os.path.basename(d).startswith(args.match)
    )
    if not runs:
        print('no runs matching %r under %s' % (args.match, args.root), file=sys.stderr)
        return 1

    print('%-42s %5s %4s %6s %7s %7s %6s  %s'
          % ('run', 'ep', 'K', 'active', 'perplex', 'maxshr', 'temp', 'head'))
    print('-' * 118)
    for run_dir in runs:
        path = checkpoint_for(run_dir, args.episode)
        if path is None:
            continue
        episode = os.path.basename(path).split('_')[0]
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        for head in ('actor_hypernet', 'value_hypernet'):
            state = checkpoint.get(head)
            if not isinstance(state, dict):
                continue
            stats = summarise_head(state, args.threshold)
            if stats is None:
                continue
            print('%-42s %5s %4d %6d %7.2f %7.3f %6.3f  %s' % (
                os.path.basename(run_dir), episode, stats['k'], stats['active'],
                stats['perplexity'], stats['max_share'], stats['temperature'],
                head.replace('_hypernet', '')))
    print()
    print('active   = prototypes holding more than %.0f%% of the gate mass' % (args.threshold * 100))
    print('perplex  = exp(entropy): the K this arm behaves as, against the K requested')
    print('maxshr   = largest single prototype share; near 1.0 is collapse to constmeta')
    return 0


if __name__ == '__main__':
    sys.exit(main())
