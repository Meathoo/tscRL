"""Tests for the learning-rate and entropy schedules.

Run inside the container:

    cd /DaRL/LibSignal && python -m unittest tests.test_annealing -v
"""

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from common.registry import Registry
from common.utils import load_config


def _agent(**overrides):
    from agent.hyperlight_ppo import HyperLightMAPPOAgent
    from tests.test_hyperlight_architecture import _fake_cityflow_world

    config, _, _ = load_config('configs/tsc/hyperlight_mappo.yml')
    config['model']['use_cuda'] = False
    config['model'].update(overrides)
    config['trainer']['episodes'] = 100
    Registry.mapping['model_mapping']['setting'] = SimpleNamespace(param=config['model'])
    Registry.mapping['trainer_mapping']['setting'] = SimpleNamespace(param=config['trainer'])
    return HyperLightMAPPOAgent(_fake_cityflow_world(), 0)


class ScheduleTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        np.random.seed(0)
        self.old_model = Registry.mapping['model_mapping'].get('setting')
        self.old_trainer = Registry.mapping['trainer_mapping'].get('setting')

    def tearDown(self):
        for mapping, saved in (('model_mapping', self.old_model),
                               ('trainer_mapping', self.old_trainer)):
            if saved is None:
                Registry.mapping[mapping].pop('setting', None)
            else:
                Registry.mapping[mapping]['setting'] = saved

    def test_default_is_off_and_touches_nothing(self):
        agent = _agent()
        before = agent.optimizer.param_groups[0]['lr']
        for _ in range(50):
            entropy_coef = agent._apply_annealing()
        self.assertEqual(agent.optimizer.param_groups[0]['lr'], before)
        self.assertEqual(entropy_coef, agent.entropy_coef)
        self.assertEqual(agent.current_schedule(), {})

    def test_learning_rate_decays_linearly_to_zero(self):
        agent = _agent(lr_anneal='linear')
        base = agent.base_learning_rate
        total = agent.total_updates

        agent._apply_annealing()
        self.assertAlmostEqual(
            agent.optimizer.param_groups[0]['lr'], base * (1.0 - 1.0 / total), places=9
        )

        agent._updates_done = total // 2 - 1
        agent._apply_annealing()
        self.assertAlmostEqual(agent.optimizer.param_groups[0]['lr'], base * 0.5, places=9)

        agent._updates_done = total - 1
        agent._apply_annealing()
        self.assertAlmostEqual(agent.optimizer.param_groups[0]['lr'], 0.0, places=12)

    def test_learning_rate_never_goes_negative_past_the_budget(self):
        """Runs overshoot their episode budget when resumed, so the schedule is
        clamped rather than allowed to invert the gradient step."""
        agent = _agent(lr_anneal='linear')
        agent._updates_done = agent.total_updates * 3
        agent._apply_annealing()
        self.assertEqual(agent.optimizer.param_groups[0]['lr'], 0.0)

    def test_entropy_decays_to_its_floor_not_to_zero(self):
        agent = _agent(entropy_anneal='linear', entropy_final_frac=0.1)
        start = agent.entropy_coef

        first = agent._apply_annealing()
        self.assertGreater(first, start * 0.9)

        agent._updates_done = agent.total_updates
        last = agent._apply_annealing()
        self.assertAlmostEqual(last, start * 0.1, places=9)
        self.assertGreater(last, 0.0)

    def test_entropy_schedule_leaves_the_learning_rate_alone(self):
        agent = _agent(entropy_anneal='linear')
        before = agent.optimizer.param_groups[0]['lr']
        agent._updates_done = agent.total_updates
        agent._apply_annealing()
        self.assertEqual(agent.optimizer.param_groups[0]['lr'], before)

    def test_total_updates_follows_the_episode_budget(self):
        agent = _agent(lr_anneal='linear')
        # 3600 steps / action_interval 10 = 360 decisions, one rollout each
        self.assertEqual(agent.total_updates, 100)

    def test_resume_continues_the_schedule_instead_of_restarting_it(self):
        """The failure this guards is silent: a resumed run would otherwise
        train its remaining episodes at the full learning rate."""
        agent = _agent(lr_anneal='linear')
        for _ in range(60):
            agent._apply_annealing()
        mid_lr = agent.optimizer.param_groups[0]['lr']
        self.assertLess(mid_lr, agent.base_learning_rate * 0.5)

        checkpoint = {'updates_done': agent._updates_done}
        fresh = _agent(lr_anneal='linear')
        fresh._updates_done = int(checkpoint['updates_done'])
        fresh._apply_annealing()
        self.assertLess(fresh.optimizer.param_groups[0]['lr'], agent.base_learning_rate * 0.5)

    def test_schedule_report_tracks_progress(self):
        agent = _agent(lr_anneal='linear', entropy_anneal='linear')
        # _apply_annealing counts this update first, so seed one short of half
        agent._updates_done = agent.total_updates // 2 - 1
        agent._apply_annealing()
        report = agent.current_schedule()
        self.assertAlmostEqual(report['progress'], 0.5, places=2)
        self.assertLess(report['lr'], agent.base_learning_rate)
        self.assertLess(report['entropy_coef'], agent.entropy_coef)

    def test_signature_records_the_schedule_only_when_enabled(self):
        plain = _agent()._architecture_signature()
        self.assertIsNone(plain['lr_anneal'])
        self.assertIsNone(plain['entropy_anneal'])

        scheduled = _agent(lr_anneal='linear', entropy_anneal='linear')
        signature = scheduled._architecture_signature()
        self.assertEqual(signature['lr_anneal'], 'linear')
        self.assertEqual(signature['entropy_anneal'], 'linear')

    def test_unknown_schedule_is_rejected(self):
        with self.assertRaises(ValueError):
            _agent(lr_anneal='cosine')


if __name__ == '__main__':
    unittest.main()
