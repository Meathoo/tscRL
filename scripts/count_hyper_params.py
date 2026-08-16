"""
Offline parameter budget for the HyperLight actor/critic hypernetworks.

Builds the same generators ``agent/hyperlight_ppo.py`` builds, but without the
simulator, so a configuration can be priced before it is queued for training.

Examples
--------
# the three methods from docs/HYPERNETWORK_COMPRESSION_METHODS.md
python scripts/count_hyper_params.py --preset compression_doc

# the chunk_size sweep, to show it interpolates towards all_weights
python scripts/count_hyper_params.py --preset chunk_sweep

# one explicit setting
python scripts/count_hyper_params.py --head_mode chunked --chunk_size 8 \
    --chunk_generator_hidden 64 --hyper_hidden 256
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.hypernetwork import build_hypernetwork  # noqa: E402

# Defaults follow configs/tsc/hyperlight_mappo.yml on cityflow16x3/7x28:
# state 32 -> 64 -> 64 -> 8 actor, 5*32=160 -> 128 -> 64 -> 1 centralized critic.
ACTOR_DIMS = [32, 64, 64, 8]
VALUE_DIMS = [160, 128, 64, 1]


def build_layout(dims):
    layout = []
    offset = 0
    for layer_idx, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
        weight_numel = out_dim * in_dim
        bias_numel = out_dim
        layout.append(
            (
                f'layer{layer_idx}',
                (out_dim, in_dim),
                offset,
                offset + weight_numel,
                offset + weight_numel + bias_numel,
            )
        )
        offset += weight_numel + bias_numel
    return layout


def target_numel(dims):
    return sum(out_dim * in_dim + out_dim for in_dim, out_dim in zip(dims[:-1], dims[1:]))


def count(head_mode, meta_dim, hyper_hidden, chunk_size, chunk_embed_dim,
          chunk_generator_hidden, actor_dims=ACTOR_DIMS, value_dims=VALUE_DIMS,
          actor_chunk_size=None, critic_chunk_size=None):
    totals = {}
    for name, dims, size in (
        ('actor', actor_dims, actor_chunk_size or chunk_size),
        ('critic', value_dims, critic_chunk_size or chunk_size),
    ):
        layout = build_layout(dims)
        net = build_hypernetwork(
            'mlp',
            meta_dim,
            list(hyper_hidden),
            target_numel(dims),
            target_layout=layout,
            head_mode=head_mode,
            chunk_size=size,
            chunk_embed_dim=chunk_embed_dim,
            chunk_generator_hidden=chunk_generator_hidden,
        )
        totals[name] = sum(param.numel() for param in net.parameters())
    totals['total'] = totals['actor'] + totals['critic']
    return totals


def film_params(meta_dim, hyper_hidden, actor_dims, value_dims):
    """FiLM emits 2 gamma/beta pairs per hidden layer instead of full weights."""
    totals = {}
    for name, dims in (('actor', actor_dims), ('critic', value_dims)):
        film_dim = 2 * sum(dims[1:-1])
        widths = [meta_dim] + list(hyper_hidden) + [film_dim]
        totals[name] = sum(
            widths[i] * widths[i + 1] + widths[i + 1] for i in range(len(widths) - 1)
        )
    totals['total'] = totals['actor'] + totals['critic']
    return totals


def row(label, totals, reference=None):
    ratio = ''
    if reference:
        ratio = f"{reference['total'] / totals['total']:.1f}x"
    print(f"{label:<34}{totals['actor']:>12,}{totals['critic']:>12,}"
          f"{totals['total']:>12,}{ratio:>10}")


def header():
    print(f"{'setting':<34}{'actor':>12}{'critic':>12}{'total':>12}{'saving':>10}")
    print('-' * 80)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--preset', choices=['compression_doc', 'chunk_sweep', 'plan'],
                        default=None)
    parser.add_argument('--head_mode', default='chunked',
                        choices=['flat', 'layerwise', 'chunked'])
    parser.add_argument('--meta_dim', type=int, default=64)
    parser.add_argument('--hyper_hidden', default='64')
    parser.add_argument('--chunk_size', type=int, default=8)
    parser.add_argument('--actor_chunk_size', type=int, default=None)
    parser.add_argument('--critic_chunk_size', type=int, default=None)
    parser.add_argument('--chunk_embed_dim', type=int, default=16)
    parser.add_argument('--chunk_generator_hidden', type=int, default=0)
    args = parser.parse_args()

    hidden = [int(x) for x in str(args.hyper_hidden).split(',')]
    all_weights = count('flat', args.meta_dim, [64], 8, 16, 0)

    if args.preset == 'compression_doc':
        header()
        row('all_weights (flat)', all_weights, all_weights)
        row('chunked c8 e16', count('chunked', 64, [64], 8, 16, 0), all_weights)
        row('FiLM', film_params(64, [64], ACTOR_DIMS, VALUE_DIMS), all_weights)
        return

    if args.preset == 'chunk_sweep':
        header()
        for size in (2, 4, 8, 16, 32, 64):
            row(f'chunked c{size} e16', count('chunked', 64, [64], size, 16, 0), all_weights)
        row('all_weights (flat)', all_weights, all_weights)
        return

    if args.preset == 'plan':
        header()
        row('all_weights (flat)', all_weights, all_weights)
        row('c8 e16 (current chunked)', count('chunked', 64, [64], 8, 16, 0), all_weights)
        row('c8 + rf_init', count('chunked', 64, [64], 8, 16, 0), all_weights)
        row('c8 + gen_hidden 64', count('chunked', 64, [64], 8, 16, 64), all_weights)
        row('c8 + hyper_hidden 256', count('chunked', 64, [256], 8, 16, 0), all_weights)
        row('c8 + hh256 + gen64', count('chunked', 64, [256], 8, 16, 64), all_weights)
        row('actor c16 / critic c4',
            count('chunked', 64, [64], 8, 16, 0, actor_chunk_size=16, critic_chunk_size=4),
            all_weights)
        return

    header()
    totals = count(args.head_mode, args.meta_dim, hidden, args.chunk_size,
                   args.chunk_embed_dim, args.chunk_generator_hidden,
                   actor_chunk_size=args.actor_chunk_size,
                   critic_chunk_size=args.critic_chunk_size)
    label = (f"{args.head_mode} c{args.actor_chunk_size or args.chunk_size}"
             f":{args.critic_chunk_size or args.chunk_size} e{args.chunk_embed_dim} "
             f"g{args.chunk_generator_hidden} hh{','.join(str(h) for h in hidden)}")
    row(label, totals, all_weights)


if __name__ == '__main__':
    main()
