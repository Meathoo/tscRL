"""Unit tests for dynamic (traffic-state) conditioning.

Run inside the container:

    cd /DaRL/LibSignal && python -m unittest dynamic.test_dynamic -v
"""

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from common.registry import Registry
from common.utils import load_config
from dynamic import (
    FEATURE_DIM,
    FEATURE_NAMES,
    RAW_DIM,
    RAW_NAMES,
    DynamicFeatureTracker,
    spec_id,
    summarize,
)


class TrackerTests(unittest.TestCase):
    def setUp(self):
        self.tracker = DynamicFeatureTracker(3, halflife_steps=10)
        self.raw = np.arange(3 * RAW_DIM, dtype=np.float32).reshape(3, RAW_DIM)

    def test_peek_and_commit_agree(self):
        """The invariant the PPO update depends on.

        remember() peeks at the next state's features; the following
        get_action() commits the very same step. If those two disagreed, the
        stored transition would describe conditioning the policy never saw.
        """
        self.tracker.step(self.raw, commit=True)
        later = self.raw + 3.0
        peeked = self.tracker.step(later, commit=False)
        committed = self.tracker.step(later, commit=True)
        np.testing.assert_array_equal(peeked, committed)

    def test_peek_does_not_move_the_state(self):
        self.tracker.step(self.raw, commit=True)
        before = self.tracker._ema.copy()
        self.tracker.step(self.raw + 10.0, commit=False)
        np.testing.assert_array_equal(before, self.tracker._ema)

    def test_first_step_seeds_instead_of_decaying_from_zero(self):
        out = self.tracker.step(self.raw, commit=True)
        queue_idx = FEATURE_NAMES.index('ema_queue')
        expected = self.raw[:, RAW_NAMES.index('queue')] / 10.0
        np.testing.assert_allclose(out[:, queue_idx], expected, rtol=1e-6)

    def test_slope_is_zero_on_the_first_step(self):
        out = self.tracker.step(self.raw, commit=True)
        self.assertTrue(np.all(out[:, FEATURE_NAMES.index('ema_queue_slope')] == 0.0))

    def test_halflife_halves_the_gap(self):
        cold = np.zeros((3, RAW_DIM), dtype=np.float32)
        self.tracker.step(cold, commit=True)
        target = np.ones((3, RAW_DIM), dtype=np.float32) * 4.0
        for _ in range(10):
            self.tracker.step(target, commit=True)
        occupancy = self.tracker._ema[:, RAW_NAMES.index('occupancy')]
        # after one half-life the remaining gap to the target is half of 4.0
        np.testing.assert_allclose(occupancy, np.full(3, 2.0), rtol=1e-3)

    def test_reset_forgets_the_trajectory(self):
        self.tracker.step(self.raw, commit=True)
        self.assertTrue(self.tracker.initialised)
        self.tracker.reset()
        self.assertFalse(self.tracker.initialised)
        reseeded = self.tracker.step(self.raw, commit=True)
        np.testing.assert_allclose(
            reseeded[:, FEATURE_NAMES.index('ema_queue')],
            self.raw[:, RAW_NAMES.index('queue')] / 10.0,
            rtol=1e-6,
        )

    def test_non_finite_input_is_sanitised(self):
        bad = self.raw.copy()
        bad[1, 0] = np.nan
        bad[2, 1] = np.inf
        out = self.tracker.step(bad, commit=True)
        self.assertTrue(np.all(np.isfinite(out)))

    def test_wrong_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            self.tracker.step(np.zeros((2, RAW_DIM), dtype=np.float32))

    def test_spec_id_pins_the_halflife(self):
        self.assertIn('hl10', spec_id(10))
        self.assertNotEqual(spec_id(10), spec_id(30))
        for name in FEATURE_NAMES:
            self.assertIn(name, spec_id(10))

    def test_summarize_mentions_every_feature(self):
        text = summarize(self.tracker.step(self.raw))
        for name in FEATURE_NAMES:
            self.assertIn(name, text)


def _agent_with_dynamic(**overrides):
    from tests.test_hyperlight_architecture import _fake_cityflow_world
    from agent.hyperlight_ppo import HyperLightMAPPOAgent

    config, _, _ = load_config('configs/tsc/hyperlight_mappo.yml')
    config['model']['use_cuda'] = False
    config['model']['dynamic_condition_enabled'] = True
    config['model'].update(overrides)
    Registry.mapping['model_mapping']['setting'] = SimpleNamespace(param=config['model'])
    Registry.mapping['trainer_mapping']['setting'] = SimpleNamespace(param=config['trainer'])
    return HyperLightMAPPOAgent(_fake_cityflow_world(), 0)


class AgentIntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        np.random.seed(0)
        self.old_model = Registry.mapping['model_mapping'].get('setting')
        self.old_trainer = Registry.mapping['trainer_mapping'].get('setting')

    def tearDown(self):
        for mapping, saved in (
            ('model_mapping', self.old_model),
            ('trainer_mapping', self.old_trainer),
        ):
            if saved is None:
                Registry.mapping[mapping].pop('setting', None)
            else:
                Registry.mapping[mapping]['setting'] = saved

    def _state(self, agent, ob, phase):
        state = agent._build_state_np(ob, phase)
        return torch.tensor(state, dtype=torch.float32).unsqueeze(0)

    def test_zero_init_leaves_meta_untouched(self):
        """A run with dynamic conditioning on must start where it would have
        started with it off, so an existing baseline stays reproducible."""
        agent = _agent_with_dynamic()
        features = agent._dynamic_advance(commit=True)
        with torch.no_grad():
            static_meta = agent._agent_meta(1, dynamic=agent._dynamic_tensor(np.zeros_like(features)))
            dynamic_meta = agent._agent_meta(1, dynamic=agent._dynamic_tensor(features))
        torch.testing.assert_close(static_meta, dynamic_meta)

    def test_meta_moves_once_the_encoder_is_trained(self):
        agent = _agent_with_dynamic()
        features = agent._dynamic_advance(commit=True)
        # emulate a trained encoder
        for layer in agent.dynamic_encoder.modules():
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.normal_(layer.weight, std=0.1)
        with torch.no_grad():
            base = agent._agent_meta(1, dynamic=agent._dynamic_tensor(np.zeros_like(features)))
            moved = agent._agent_meta(1, dynamic=agent._dynamic_tensor(features * 3.0))
        self.assertFalse(torch.allclose(base, moved))

    def test_missing_features_raise_instead_of_diverging_silently(self):
        """The failure this guards is invisible at runtime: without the
        recorded features the update would condition on something else and the
        PPO ratio would be quietly wrong."""
        agent = _agent_with_dynamic()
        ob, phase = agent.get_ob(), agent.get_phase()
        state = self._state(agent, ob, phase)
        with self.assertRaises(RuntimeError) as caught:
            agent._policy_value(state)
        self.assertIn('log-probabilities', str(caught.exception))

    def test_update_replays_the_features_recorded_at_rollout(self):
        agent = _agent_with_dynamic()
        agent.reset()
        ob, phase = agent.get_ob(), agent.get_phase()
        actions = agent.get_action(ob, phase)
        probs = agent.get_action_prob(ob, phase)

        rewards = np.zeros((agent.sub_agents,), dtype=np.float32)
        agent.remember(ob, phase, actions, probs, rewards, ob, phase, False, 'k0')

        stored_dynamic = agent.rollout_buffer[-1][9]
        state = self._state(agent, ob, phase)
        with torch.no_grad():
            logits, _ = agent._policy_value(
                state,
                dynamic=agent._dynamic_tensor(stored_dynamic),
            )
            replayed = torch.softmax(logits.squeeze(0), dim=-1)
        torch.testing.assert_close(replayed, probs, rtol=1e-5, atol=1e-6)

    def test_stored_next_features_match_the_next_decision(self):
        agent = _agent_with_dynamic()
        agent.reset()
        ob, phase = agent.get_ob(), agent.get_phase()
        actions = agent.get_action(ob, phase)
        probs = agent.get_action_prob(ob, phase)
        rewards = np.zeros((agent.sub_agents,), dtype=np.float32)
        agent.remember(ob, phase, actions, probs, rewards, ob, phase, False, 'k0')
        stored_next = agent.rollout_buffer[-1][10]

        # the world has not moved in this fixture, so the next decision must
        # commit exactly what remember() peeked at
        agent.get_action(ob, phase)
        np.testing.assert_array_equal(stored_next, agent._dynamic_current)

    def test_reset_clears_the_tracker_between_episodes(self):
        agent = _agent_with_dynamic()
        agent.get_action(agent.get_ob(), agent.get_phase())
        self.assertTrue(agent.dynamic_tracker.initialised)
        agent.reset()
        self.assertFalse(agent.dynamic_tracker.initialised)
        self.assertIsNone(agent._dynamic_current)

    def test_encoder_is_trained_and_signature_records_the_spec(self):
        agent = _agent_with_dynamic(dynamic_ema_halflife=30)
        params = {id(p) for p in agent._optimizer_parameters()}
        for param in agent.dynamic_encoder.parameters():
            self.assertIn(id(param), params)
        signature = agent._architecture_signature()
        self.assertIn('hl30', signature['dynamic_spec'])

    def test_raw_features_have_the_declared_width(self):
        agent = _agent_with_dynamic()
        raw = agent._dynamic_raw()
        self.assertEqual(raw.shape, (agent.sub_agents, RAW_DIM))
        self.assertTrue(np.all(np.isfinite(raw)))
        features = agent._dynamic_advance(commit=True)
        self.assertEqual(features.shape, (agent.sub_agents, FEATURE_DIM))


if __name__ == '__main__':
    unittest.main()
