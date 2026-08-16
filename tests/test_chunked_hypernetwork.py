import unittest

import torch

from agent.hypernetwork import ChunkedHyperNetwork, build_hypernetwork


def _layout(dims):
    layout = []
    offset = 0
    for layer_idx, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
        layout.append(
            (
                f'layer{layer_idx}',
                (out_dim, in_dim),
                offset,
                offset + out_dim * in_dim,
                offset + out_dim * in_dim + out_dim,
            )
        )
        offset += out_dim * in_dim + out_dim
    return layout


def _build(generator_hidden, dims=(32, 64), chunk_size=8, meta_dim=64):
    torch.manual_seed(0)
    layout = _layout(list(dims))
    return ChunkedHyperNetwork(
        meta_dim,
        [64],
        layout[-1][-1],
        target_layout=layout,
        chunk_size=chunk_size,
        chunk_embed_dim=16,
        generator_hidden=generator_hidden,
    )


class ChunkedGeneratorConditioningTests(unittest.TestCase):
    """
    The single-Linear generator emits ``W_h h + W_c code_j + b`` for every chunk,
    so the agent-dependent part of a generated weight matrix is one row-block
    tiled down the matrix. A hidden layer after the concatenation removes that
    constraint; these tests pin both behaviours down.
    """

    def _agent_variation_blocks(self, net, out_dim=64, in_dim=32, chunk_size=8):
        torch.manual_seed(1)
        meta = torch.randn(2, 64)
        theta = net(meta)
        delta = (theta[0] - theta[1])[: out_dim * in_dim].view(out_dim, in_dim)
        return delta.split(chunk_size, dim=0)

    def test_additive_generator_tiles_agent_variation(self):
        blocks = self._agent_variation_blocks(_build(0))
        for block in blocks[1:]:
            self.assertTrue(torch.allclose(block, blocks[0], atol=1e-6))

    def test_hidden_generator_breaks_the_tiling(self):
        blocks = self._agent_variation_blocks(_build(64))
        differences = [
            (block - blocks[0]).abs().max().item() for block in blocks[1:]
        ]
        self.assertGreater(min(differences), 1e-3)

    def test_hidden_generator_is_not_more_expensive_on_the_real_actor(self):
        # The wide output head reads from 64 units instead of trunk+code=80, which
        # pays for the extra layer once a target layer is wide enough. Holds for the
        # real 32-64-64-8 actor, not necessarily for a single narrow layer.
        actor = (32, 64, 64, 8)
        additive = sum(p.numel() for p in _build(0, dims=actor).parameters())
        hidden = sum(p.numel() for p in _build(64, dims=actor).parameters())
        self.assertLessEqual(hidden, additive)

    def test_output_dim_and_shapes_are_unchanged(self):
        for generator_hidden in (0, 64):
            net = _build(generator_hidden, dims=(160, 128, 64, 1))
            theta = net(torch.randn(3, 5, 64))
            self.assertEqual(theta.shape, (3, 5, net.output_dim))

    def test_flat_head_honours_rf_init(self):
        """head_mode=flat is the all_weights baseline; rf_init used to skip it entirely."""
        layout = _layout([32, 64, 64, 8])
        built = {}
        for rf in (False, True):
            torch.manual_seed(0)
            built[rf] = build_hypernetwork(
                'mlp', 64, [64], layout[-1][-1],
                target_layout=layout, head_mode='flat', rf_init=rf,
            )
        plain = dict(built[False].named_parameters())
        scaled = dict(built[True].named_parameters())
        self.assertEqual(sorted(plain), sorted(scaled))
        self.assertFalse(
            all(torch.equal(plain[k], scaled[k]) for k in plain),
            'rf_init must change the flat head initialization',
        )
        self.assertEqual(
            sum(p.numel() for p in built[True].parameters()),
            sum(p.numel() for p in built[False].parameters()),
            'rf_init must not change the parameter count',
        )

    def test_rf_init_applies_to_the_output_layer(self):
        layout = _layout([32, 64])
        net = build_hypernetwork(
            'mlp',
            64,
            [64],
            layout[-1][-1],
            target_layout=layout,
            head_mode='chunked',
            chunk_size=8,
            chunk_embed_dim=16,
            chunk_generator_hidden=64,
            rf_init=True,
        )
        output_layer = net.generators[0][-1]
        self.assertTrue(torch.any(output_layer.bias != 0))


if __name__ == '__main__':
    unittest.main()
