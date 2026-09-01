"""Tests for the K-prototype factorized head.

Run inside the container:

    cd /DaRL/LibSignal && python -m unittest tests.test_prototype_hypernetwork -v

The load-bearing tests are the two endpoints. K=1 has to reproduce constmeta
and K=0 has to construct nothing, because the whole claim of this arm is that
it sits on an axis whose two ends are already measured -- if the ends do not
land where the existing results say they land, the middle means nothing.
"""

import unittest

import torch

from agent.hypernetwork import PrototypeHyperNetwork, build_hypernetwork


META_DIM = 64
HIDDEN = [64]
# The real actor target: 32 -> 64 -> 64 -> 8, which is 6792 parameters.
ACTOR_OUT = 6792
ACTOR_LAYOUT = [
    ('fc1', (64, 32), 0, 2048, 2112),
    ('fc2', (64, 64), 2112, 6208, 6272),
    ('fc3', (8, 64), 6272, 6784, 6792),
]


def _build(num_prototypes, seed=0, **kwargs):
    torch.manual_seed(seed)
    return build_hypernetwork(
        'mlp', META_DIM, HIDDEN, ACTOR_OUT,
        num_prototypes=num_prototypes, **kwargs
    )


class PrototypeHeadTest(unittest.TestCase):

    def test_zero_prototypes_returns_the_plain_head_untouched(self):
        """K=0 must not wrap, or every existing arm changes its init order."""
        head = _build(0)
        self.assertNotIsInstance(head, PrototypeHyperNetwork)
        # The all_weights actor head, straight off the checkpoint tables in
        # docs/HYPERNETWORK_COMPRESSION_METHODS.md sec 6.
        self.assertEqual(sum(p.numel() for p in head.parameters()), 445640)

    def test_k_one_is_constmeta_by_construction(self):
        """Every intersection gets the same parameters, whatever the meta.

        This is the endpoint that has to match (q): a hypernetwork with no
        content, which lost to a plain shared MLP by 61.7s on Ingolstadt21.
        """
        head = _build(1)
        meta = torch.randn(2, 21, META_DIM)
        theta = head(meta)
        self.assertEqual((theta - theta[:, :1]).abs().max().item(), 0.0)

    def test_varying_meta_gives_each_intersection_its_own_parameters(self):
        head = _build(8)
        meta = torch.randn(2, 21, META_DIM)
        theta = head(meta)
        self.assertGreater((theta - theta[:, :1]).abs().max().item(), 0.0)

    def test_constant_meta_collapses_at_any_k(self):
        """A constant conditioning vector cannot be rescued by more prototypes.

        Worth pinning: on the CityFlow grids the structural contract is nearly
        constant (10 of 12 features), so this arm is expected to be null there
        for the same reason everything else has been. The gate reads the meta
        and nothing else, so that expectation is structural, not empirical.
        """
        head = _build(8)
        meta = torch.ones(2, 21, META_DIM)
        theta = head(meta)
        self.assertEqual((theta - theta[:, :1]).abs().max().item(), 0.0)

    def test_mixing_weights_are_a_simplex(self):
        head = _build(8)
        alpha = head.mixing_weights(torch.randn(3, 21, META_DIM))
        self.assertEqual(tuple(alpha.shape), (3, 21, 8))
        self.assertTrue(torch.allclose(alpha.sum(-1), torch.ones(3, 21), atol=1e-6))
        self.assertTrue((alpha >= 0).all())

    def test_parameter_cost_is_independent_of_agent_count(self):
        """K codes and one gate, both shaped by meta_dim -- never by N."""
        head = _build(8)
        expected = 445640 + 8 * META_DIM + (META_DIM * 64 + 64 + 64 * 8 + 8)
        self.assertEqual(sum(p.numel() for p in head.parameters()), expected)

    def test_generator_runs_k_times_not_once_per_agent(self):
        """The compute argument, asserted rather than asserted in prose."""
        head = _build(4)
        calls = []
        inner_forward = head.inner.forward

        def counting_forward(x):
            calls.append(x.shape)
            return inner_forward(x)

        head.inner.forward = counting_forward
        head(torch.randn(8, 196, META_DIM))
        self.assertEqual(len(calls), 1)
        self.assertEqual(tuple(calls[0]), (4, META_DIM))

    def test_composes_with_the_chunked_head(self):
        head = _build(8, head_mode='chunked', chunk_size=8, chunk_embed_dim=16,
                      target_layout=ACTOR_LAYOUT)
        theta = head(torch.randn(2, 21, META_DIM))
        self.assertEqual(tuple(theta.shape), (2, 21, ACTOR_OUT))

    def test_composes_with_the_layerwise_head(self):
        head = _build(8, head_mode='layerwise', target_layout=ACTOR_LAYOUT)
        theta = head(torch.randn(2, 21, META_DIM))
        self.assertEqual(tuple(theta.shape), (2, 21, ACTOR_OUT))

    def test_frozen_gate_freezes_only_the_gate(self):
        head = _build(8, prototype_gate_frozen=True)
        self.assertFalse(any(p.requires_grad for p in head.gate.parameters()))
        self.assertTrue(head.prototypes.requires_grad)

    def test_temperature_travels_in_the_state_dict(self):
        """A resumed run must not silently restart the anneal."""
        head = _build(8)
        head.set_temperature(0.37)
        self.assertIn('temperature', head.state_dict())
        restored = _build(8, seed=1)
        restored.load_state_dict(head.state_dict())
        self.assertAlmostEqual(float(restored.temperature), 0.37, places=6)

    def test_temperature_sharpens_the_gate(self):
        head = _build(8)
        meta = torch.randn(4, 21, META_DIM)
        head.set_temperature(4.0)
        hot = head.mixing_weights(meta).max(-1).values.mean()
        head.set_temperature(0.1)
        cold = head.mixing_weights(meta).max(-1).values.mean()
        self.assertGreater(cold.item(), hot.item())

    def test_reinit_dead_redraws_only_unused_prototypes(self):
        head = _build(8)
        with torch.no_grad():
            head.usage.zero_()
            head.usage[3] = 1.0
        live = head.prototypes[3].clone()
        redrawn = head.reinit_dead()
        self.assertEqual(redrawn, 7)
        self.assertTrue(torch.equal(head.prototypes[3], live))

    def test_reinit_dead_is_a_noop_when_all_prototypes_are_used(self):
        head = _build(8)
        with torch.no_grad():
            head.usage.fill_(1.0)
        before = head.prototypes.clone()
        self.assertEqual(head.reinit_dead(), 0)
        self.assertTrue(torch.equal(head.prototypes, before))

    def test_reinit_dead_never_redraws_everything(self):
        """All-dead means the usage stats are stale, not that K is wrong."""
        head = _build(8)
        with torch.no_grad():
            head.usage.zero_()
        before = head.prototypes.clone()
        self.assertEqual(head.reinit_dead(), 0)
        self.assertTrue(torch.equal(head.prototypes, before))

    def test_diagnostics_report_collapse(self):
        head = _build(8)
        with torch.no_grad():
            head.usage.zero_()
            head.usage[0] = 1.0
        stats = head.diagnostics()
        self.assertEqual(stats['proto_active'], 1.0)
        self.assertAlmostEqual(stats['proto_max_share'], 1.0, places=5)
        self.assertAlmostEqual(stats['proto_entropy'], 0.0, places=5)

    def test_gradients_reach_prototypes_and_gate(self):
        head = _build(4)
        head(torch.randn(2, 5, META_DIM)).sum().backward()
        self.assertIsNotNone(head.prototypes.grad)
        self.assertGreater(head.prototypes.grad.abs().sum().item(), 0.0)
        gate_grads = [p.grad for p in head.gate.parameters() if p.grad is not None]
        self.assertTrue(gate_grads)

    def test_rejects_non_positive_k(self):
        with self.assertRaises(ValueError):
            PrototypeHyperNetwork(torch.nn.Identity(), META_DIM, ACTOR_OUT, 0)


if __name__ == '__main__':
    unittest.main()
