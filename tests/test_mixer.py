"""Tests for the regime-quantized credit-assignment mixer.

Run inside the container:

    cd /DaRL/LibSignal && python -m unittest tests.test_mixer -v

The two that carry weight are the monotonicity (no agent's advantage may be
sign-flipped relative to the joint one, which is what makes A_i = w_i * A_tot
a decomposition rather than a reweighting) and the N-independence (QMIX's mixer
is shaped [N, hidden] and cannot cross a road network; this one must not be).
"""

import unittest

import torch

from agent.mixer import RegimeMixer, UniformMixer


STATE_DIM = 32
COND_DIM = 64


def _build(seed=0, **kwargs):
    torch.manual_seed(seed)
    return RegimeMixer(STATE_DIM, COND_DIM, **kwargs)


def _inputs(batch=6, agents=21):
    torch.manual_seed(1)
    return torch.randn(batch, agents, STATE_DIM), torch.randn(agents, COND_DIM)


class RegimeMixerTest(unittest.TestCase):

    def test_starts_as_the_plain_sum_of_local_values(self):
        """Zero-init heads mean w = 1 and b = 0, so V_tot begins as sum_i V_i.

        That makes the starting point shared-reward MAPPO, a recognisable
        algorithm, rather than an arbitrary configuration.
        """
        mixer = _build()
        state, cond = _inputs()
        weights, bias, _ = mixer(state, cond)
        self.assertTrue(torch.allclose(weights, torch.ones_like(weights), atol=1e-6))
        self.assertTrue(torch.allclose(bias, torch.zeros_like(bias), atol=1e-6))

    def test_weights_stay_positive_under_random_parameters(self):
        """Monotonicity is the property A_i = w_i * A_tot depends on."""
        mixer = _build()
        with torch.no_grad():
            for param in mixer.parameters():
                param.add_(torch.randn_like(param) * 3.0)
        state, cond = _inputs()
        weights, _, _ = mixer(state, cond)
        self.assertTrue((weights > 0).all())

    def test_parameter_count_is_independent_of_agent_count(self):
        """QMIX's mixer is shaped by N. This one must not be."""
        mixer = _build()
        before = sum(p.numel() for p in mixer.parameters())
        for agents in (3, 21, 196):
            state, cond = _inputs(agents=agents)
            weights, bias, _ = mixer(state, cond)
            self.assertEqual(tuple(weights.shape), (6, agents))
            self.assertEqual(tuple(bias.shape), (6,))
        self.assertEqual(sum(p.numel() for p in mixer.parameters()), before)

    def test_is_permutation_equivariant_in_the_agents(self):
        mixer = _build()
        with torch.no_grad():
            for param in mixer.parameters():
                param.add_(torch.randn_like(param) * 0.5)
        state, cond = _inputs()
        perm = torch.randperm(state.shape[1])
        w_plain, bias_plain, _ = mixer(state, cond)
        w_perm, bias_perm, _ = mixer(state[:, perm], cond[perm])
        self.assertTrue(torch.allclose(w_plain[:, perm], w_perm, atol=1e-5))
        self.assertTrue(torch.allclose(bias_plain, bias_perm, atol=1e-5))

    def test_quantized_code_is_piecewise_constant(self):
        """The property (h) could not have: while the regime holds, the
        generated weights do not move at all."""
        mixer = _build(num_regimes=2)
        state, cond = _inputs(batch=64)
        mixer.eval()
        z, _, index = mixer.encode(state)
        for regime in index.unique():
            rows = z[index == regime]
            self.assertTrue(torch.allclose(rows, rows[:1], atol=1e-6))

    def test_continuous_mode_is_not_piecewise_constant(self):
        """The B3 arm: without quantization the code moves every step."""
        mixer = _build(quantize=False)
        state, cond = _inputs(batch=32)
        z, vq_loss, index = mixer.encode(state)
        self.assertIsNone(index)
        self.assertEqual(float(vq_loss), 0.0)
        self.assertGreater((z - z[:1]).abs().max().item(), 0.0)

    def test_straight_through_gradient_reaches_the_encoder(self):
        mixer = _build()
        state, cond = _inputs()
        weights, bias, vq = mixer(state, cond)
        (weights.sum() + bias.sum() + vq).backward()
        encoder_grads = [
            p.grad for p in mixer.regime_encoder.parameters() if p.grad is not None
        ]
        self.assertTrue(encoder_grads)
        self.assertGreater(sum(g.abs().sum().item() for g in encoder_grads), 0.0)

    def test_vq_loss_is_zero_only_when_the_code_sits_on_a_prototype(self):
        mixer = _build()
        state, cond = _inputs()
        _, vq_loss, _ = mixer.encode(state)
        self.assertGreater(float(vq_loss), 0.0)

    def test_diagnostics_report_codebook_collapse(self):
        mixer = _build(num_regimes=8)
        with torch.no_grad():
            mixer.usage.zero_()
            mixer.usage[2] = 1.0
        stats = mixer.diagnostics()
        self.assertEqual(stats['regime_active'], 1.0)
        self.assertAlmostEqual(stats['regime_perplexity'], 1.0, places=4)

    def test_reinit_dead_redraws_only_unused_regimes(self):
        mixer = _build(num_regimes=8)
        with torch.no_grad():
            mixer.usage.zero_()
            mixer.usage[5] = 1.0
        live = mixer.codebook[5].clone()
        self.assertEqual(mixer.reinit_dead(), 7)
        self.assertTrue(torch.equal(mixer.codebook[5], live))

    def test_reinit_dead_never_redraws_everything(self):
        mixer = _build(num_regimes=8)
        with torch.no_grad():
            mixer.usage.zero_()
        before = mixer.codebook.clone()
        self.assertEqual(mixer.reinit_dead(), 0)
        self.assertTrue(torch.equal(mixer.codebook, before))

    def test_usage_and_codebook_travel_in_the_state_dict(self):
        mixer = _build(num_regimes=8)
        state, cond = _inputs()
        mixer(state, cond)
        restored = _build(seed=7, num_regimes=8)
        restored.load_state_dict(mixer.state_dict())
        self.assertTrue(torch.equal(restored.usage, mixer.usage))
        self.assertTrue(torch.equal(restored.codebook, mixer.codebook))

    def test_rejects_non_positive_k(self):
        with self.assertRaises(ValueError):
            RegimeMixer(STATE_DIM, COND_DIM, num_regimes=0)


class UniformMixerTest(unittest.TestCase):

    def test_is_exactly_weight_one_and_bias_zero(self):
        mixer = UniformMixer()
        state, cond = _inputs()
        weights, bias, vq = mixer(state, cond)
        self.assertTrue(torch.equal(weights, torch.ones_like(weights)))
        self.assertTrue(torch.equal(bias, torch.zeros_like(bias)))
        self.assertEqual(float(vq), 0.0)

    def test_has_no_parameters(self):
        self.assertEqual(sum(p.numel() for p in UniformMixer().parameters()), 0)


if __name__ == '__main__':
    unittest.main()
