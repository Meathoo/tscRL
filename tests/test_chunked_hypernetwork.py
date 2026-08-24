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


class ChunkedRFInitModeTests(unittest.TestCase):
    """
    ``rf_init`` on a chunked head has to put the target-layer init somewhere every
    chunk can read it. ``shared`` uses the generator bias, which all chunks share,
    so the generated matrix starts as one block tiled down the rows and only the
    random code path breaks the tie. ``per_chunk`` slices one full-size init
    across the chunks and routes it through the code parameters instead.
    """

    CRITIC_LAYER = (160, 128)   # 128x160, 16 chunks at chunk_size 8

    @staticmethod
    def _build(head_mode='chunked', dims=CRITIC_LAYER, rf_init=True,
               chunk_rf_mode='shared', chunk_embed_dim=16,
               chunk_generator_hidden=0, chunk_size=8):
        torch.manual_seed(0)
        layout = _layout(list(dims))
        net = build_hypernetwork(
            'mlp', 64, [64], layout[-1][-1],
            target_layout=layout,
            head_mode=head_mode,
            rf_init=rf_init,
            chunk_size=chunk_size,
            chunk_embed_dim=chunk_embed_dim,
            chunk_generator_hidden=chunk_generator_hidden,
            chunk_rf_mode=chunk_rf_mode,
        )
        return net, layout

    @staticmethod
    def _generated_weight(net, layout, layer_idx=0):
        torch.manual_seed(1)
        theta = net(torch.randn(1, 64))
        _, (out_dim, in_dim), weight_start, bias_start, _ = layout[layer_idx]
        return theta[0, weight_start:bias_start].reshape(out_dim, in_dim)

    @classmethod
    def _effective_rank(cls, net, layout, layer_idx=0):
        """Participation ratio of the spectrum: how many directions actually carry
        the matrix, as opposed to how many are merely nonzero."""
        singular = torch.linalg.svdvals(cls._generated_weight(net, layout, layer_idx))
        return (singular.sum() ** 2 / singular.pow(2).sum()).item()

    def test_shared_mode_starts_the_target_layer_rank_deficient(self):
        flat = self._effective_rank(*self._build(head_mode='flat'))
        shared = self._effective_rank(*self._build(chunk_rf_mode='shared'))
        self.assertGreater(flat, 0.95 * 128)
        self.assertLess(shared, 0.25 * flat)

    def test_per_chunk_restores_the_flat_head_conditioning(self):
        flat = self._effective_rank(*self._build(head_mode='flat'))
        per_chunk = self._effective_rank(*self._build(chunk_rf_mode='per_chunk'))
        self.assertGreater(per_chunk, 0.95 * flat)

    def test_per_chunk_restores_conditioning_with_a_hidden_generator(self):
        flat = self._effective_rank(*self._build(head_mode='flat'))
        for mode, bound in (('shared', 0.25), ('per_chunk', 0.95)):
            net, layout = self._build(chunk_rf_mode=mode, chunk_generator_hidden=64)
            rank = self._effective_rank(net, layout)
            if mode == 'shared':
                self.assertLess(rank, bound * flat)
            else:
                self.assertGreater(rank, bound * flat)

    def test_per_chunk_gives_every_chunk_a_different_block(self):
        """Under shared the row-blocks start as one block plus a small deviation."""
        ratios = {}
        for mode in ('shared', 'per_chunk'):
            weight = self._generated_weight(*self._build(chunk_rf_mode=mode))
            blocks = weight.split(8, dim=0)
            spread = max((b - blocks[0]).abs().max().item() for b in blocks[1:])
            ratios[mode] = spread / blocks[0].abs().max().item()
        self.assertLess(ratios['shared'], 0.5)
        self.assertGreater(ratios['per_chunk'], 1.0)

    def test_per_chunk_costs_no_parameters(self):
        for generator_hidden in (0, 64):
            counts = {
                mode: sum(p.numel() for p in self._build(
                    chunk_rf_mode=mode, chunk_generator_hidden=generator_hidden,
                    dims=(160, 128, 64, 1))[0].parameters())
                for mode in ('shared', 'per_chunk')
            }
            self.assertEqual(counts['shared'], counts['per_chunk'])

    def test_per_chunk_still_tiles_the_agent_variation_without_a_hidden_generator(self):
        """per_chunk fixes the initialization, not the additive generator's tiling
        symmetry -- that is what chunk_generator_hidden is for. Both can be on."""
        net, layout = self._build(chunk_rf_mode='per_chunk')
        torch.manual_seed(1)
        meta = torch.randn(2, 64)
        theta = net(meta)
        _, (out_dim, in_dim), weight_start, bias_start, _ = layout[0]
        delta = (theta[0] - theta[1])[weight_start:bias_start].view(out_dim, in_dim)
        blocks = delta.split(8, dim=0)
        for block in blocks[1:]:
            self.assertTrue(torch.allclose(block, blocks[0], atol=1e-6))

    def test_per_chunk_needs_a_code_dimension_per_chunk(self):
        with self.assertRaises(ValueError) as caught:
            self._build(chunk_rf_mode='per_chunk', chunk_embed_dim=8)
        self.assertIn('chunk_embed_dim', str(caught.exception))
        self._build(chunk_rf_mode='per_chunk', chunk_embed_dim=16)   # 16 chunks, exactly enough

    def test_per_chunk_needs_a_generator_hidden_unit_per_chunk(self):
        with self.assertRaises(ValueError) as caught:
            self._build(chunk_rf_mode='per_chunk', chunk_generator_hidden=8)
        self.assertIn('generator_hidden', str(caught.exception))

    def test_per_chunk_without_rf_init_is_rejected_rather_than_ignored(self):
        with self.assertRaises(ValueError) as caught:
            self._build(chunk_rf_mode='per_chunk', rf_init=False)
        self.assertIn('hyper_rf_init', str(caught.exception))

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            self._build(chunk_rf_mode='per-chunk')

    def test_other_head_modes_ignore_the_mode(self):
        for head_mode in ('flat', 'layerwise'):
            self._build(head_mode=head_mode, chunk_rf_mode='per_chunk')


if __name__ == '__main__':
    unittest.main()
