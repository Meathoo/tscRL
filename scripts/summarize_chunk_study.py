"""
Summarize the chunked-hypernetwork study runs launched by scripts/chunk_study.sh.

Discovers every ``<tag>_<net>_seed<k>`` run directory for a network, merges the
per-attempt DTL logs the way avg_compare.py does (resumed runs leave one log per
attempt), and reports TEST travel time per config with a training-collapse check.

    python scripts/summarize_chunk_study.py --network cityflow4x4
    python scripts/summarize_chunk_study.py --network cityflow7x28 --statistic tail5
"""

import argparse
import os
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from avg_compare import load_log_dir_merged  # noqa: E402

RUN_RE = re.compile(r'^(?P<tag>.+)_(?P<net>[^_]+)_seed(?P<seed>\d+)$')


def run_dirs(root: Path):
    if not root.is_dir():
        return []
    found = []
    for path in sorted(root.iterdir()):
        match = RUN_RE.match(path.name)
        if path.is_dir() and match:
            found.append((match.group('tag'), int(match.group('seed')), path))
    return found


def dir_size_mb(path: Path):
    if not path.is_dir():
        return None
    total = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
    return total / (1024 * 1024)


def collapsed(records, run_length=3):
    """A dead policy logs loss exactly 0.0 for every remaining episode."""
    streak = 0
    for record in records:
        if record.mode != 'TRAIN':
            continue
        if record.loss == 0.0:
            streak += 1
            if streak >= run_length:
                return True
        else:
            streak = 0
    return False


def summarize_seed(path: Path, statistic: str, tail: int):
    records = load_log_dir_merged(path)
    test = [r for r in records if r.mode == 'TEST']
    train = [r for r in records if r.mode == 'TRAIN']
    if not test:
        return None
    test.sort(key=lambda r: r.episode)
    if statistic == 'last':
        value = test[-1].travel_time
    elif statistic == 'tail5':
        value = statistics.fmean([r.travel_time for r in test[-tail:]])
    elif statistic == 'best':
        value = min(r.travel_time for r in test)
    else:
        raise ValueError(statistic)
    return {
        'value': value,
        'episodes': max((r.episode for r in train), default=0),
        'collapsed': collapsed(records),
        'size_mb': dir_size_mb(path / 'model'),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', default='cityflow4x4')
    parser.add_argument('--agent', default='hyperlight_mappo')
    # tail5 by default. Evaluation is deterministic -- the same checkpoint
    # scores the same travel time every time -- so the spread along a TEST
    # curve is the policy genuinely moving between episodes, not measurement
    # noise. That makes `last` one draw from that movement, and `best` a
    # minimum over ~50 correlated draws whose optimistic bias grows with how
    # much an arm oscillates, which flatters exactly the unstable arms.
    # Averaging the tail is the only one of the three that estimates what the
    # policy typically does late in training.
    parser.add_argument('--statistic', default='tail5', choices=['last', 'tail5', 'best'])
    parser.add_argument('--tail', type=int, default=10, help='TEST points averaged by tail5')
    parser.add_argument('--root', default='data/output_data/tsc')
    parser.add_argument('--world', default='cityflow', choices=['cityflow', 'sumo'],
                        help='run.py writes into <world>_<agent>/<network>, so the '
                             'Ingolstadt runs live under sumo_hyperlight_mappo')
    parser.add_argument('--min-episodes', type=int, default=249,
                        help='seeds below this episode count are reported as in-progress '
                             'and excluded from the mean, so a half-trained run cannot '
                             'quietly drag a config average around')
    args = parser.parse_args()

    root = Path(args.root) / f'{args.world}_{args.agent}' / args.network
    per_tag = defaultdict(list)
    for tag, seed, path in run_dirs(root):
        summary = summarize_seed(path, args.statistic, args.tail)
        if summary is not None:
            summary['seed'] = seed
            per_tag[tag].append(summary)

    if not per_tag:
        print(f'no runs found under {root}')
        return

    print(f'{args.network}  TEST travel time ({args.statistic}), lower is better')
    print(f"{'config':<14}{'seeds':>6}{'mean':>10}{'std':>8}{'per-seed':>34}"
          f"{'ep':>6}{'ckpt MB':>9}  flags")
    print('-' * 100)
    baseline = None
    pending = []
    for tag in sorted(per_tag, key=lambda t: (t != 'aw', t)):
        runs = sorted(per_tag[tag], key=lambda r: r['seed'])
        done = [r for r in runs if r['episodes'] >= args.min_episodes]
        for run in runs:
            if run not in done:
                pending.append(f"{tag} seed{run['seed']} @ep{run['episodes']}"
                               f" ({run['value']:.1f})")
        if not done:
            continue
        values = [r['value'] for r in done]
        mean = statistics.fmean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        if tag == 'aw':
            baseline = mean
        per_seed = ' '.join(f'{v:.1f}' for v in values)
        episodes = min(r['episodes'] for r in done)
        size = max((r['size_mb'] or 0.0) for r in done)
        flags = []
        if any(r['collapsed'] for r in done):
            flags.append('COLLAPSE:' + ','.join(
                str(r['seed']) for r in done if r['collapsed']))
        if baseline is not None and tag != 'aw':
            flags.append(f'vs aw {mean - baseline:+.1f}')
        print(f'{tag:<14}{len(values):>6}{mean:>10.2f}{std:>8.2f}{per_seed:>34}'
              f'{episodes:>6}{size:>9.1f}  {" ".join(flags)}')

    if pending:
        print('\nin progress (excluded from the means above):')
        for item in pending:
            print(f'  {item}')


if __name__ == '__main__':
    main()
