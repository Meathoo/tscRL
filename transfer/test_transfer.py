"""Unit tests for the cross-network transfer package.

Run inside the container:

    cd /DaRL/LibSignal && python -m unittest transfer.test_transfer -v
"""

import os
import tempfile
import unittest

import numpy as np
import torch
import torch.nn as nn

from transfer import (
    FEATURE_DIM,
    FEATURE_NAMES,
    SPEC_VERSION,
    TransferError,
    build_structural_features,
    format_report,
    load_for_transfer,
    spec_id,
    validate_transfer_architecture,
)


def _road(lanes):
    return {'lanes': list(range(lanes))}


class _Intersection:
    def __init__(self, inter_id, in_lanes, out_lanes, phases, neighbors):
        self.id = inter_id
        # One road per approach; 3 lanes each is the CityFlow grid layout.
        self.in_roads = [_road(3) for _ in range(in_lanes // 3)]
        self.out_roads = [_road(3) for _ in range(out_lanes // 3)]
        self.phases = list(range(phases))
        self.startlanes = list(range(in_lanes))
        self.neighbors = neighbors


def _features(intersections):
    return build_structural_features(
        intersections,
        road_lane_count=lambda road: len(road.get('lanes', [])),
        neighbor_intersections=lambda inter: inter.neighbors,
    )


def _zscore(raw):
    """Mirror of the legacy ``_normalize_topology_features`` (per-network)."""
    mean = raw.mean(axis=0, keepdims=True)
    std = raw.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (raw - mean) / std


class StructuralFeatureTests(unittest.TestCase):
    def test_spec_shape_matches_names(self):
        self.assertEqual(len(FEATURE_NAMES), FEATURE_DIM)
        self.assertIn(f'structural_v{SPEC_VERSION}', spec_id())
        for name in FEATURE_NAMES:
            self.assertIn(name, spec_id())

    def test_same_intersection_gets_same_vector_in_different_networks(self):
        """The property the whole transfer story depends on.

        A "12 incoming lanes / 9 phases / 4 controlled neighbours" intersection
        must map to the same conditioning vector no matter what the rest of the
        network looks like.
        """
        probe = lambda: _Intersection('probe', 12, 12, 9, {'a', 'b', 'c', 'd'})

        small = [probe()] + [_Intersection('x', 6, 6, 4, {'a'}) for _ in range(3)]
        large = [probe()] + [
            _Intersection(f'y{i}', 15, 15, 12, {'a', 'b'}) for i in range(40)
        ]

        small_scaled, _, small_raw = _features(small)
        large_scaled, _, large_raw = _features(large)

        np.testing.assert_allclose(small_scaled[0], large_scaled[0], rtol=0, atol=0)

        # ...and the legacy per-network z-score does NOT have that property,
        # which is exactly why this module exists.
        legacy_small = _zscore(small_raw)
        legacy_large = _zscore(large_raw)
        self.assertFalse(
            np.allclose(legacy_small[0], legacy_large[0]),
            'legacy z-score unexpectedly stable; the regression this guards is gone',
        )

    def test_features_are_finite_and_scaled(self):
        scaled, names, raw = _features(
            [_Intersection('a', 12, 12, 9, {'b'}), _Intersection('b', 3, 3, 2, set())]
        )
        self.assertEqual(scaled.shape, (2, FEATURE_DIM))
        self.assertEqual(names, list(FEATURE_NAMES))
        self.assertTrue(np.isfinite(scaled).all())
        self.assertTrue((np.abs(scaled) <= 3.0).all(), scaled)
        # raw keeps the physical quantity for logging
        self.assertEqual(raw[0, FEATURE_NAMES.index('in_lane_count')], 12.0)

    def test_single_intersection_network_does_not_collapse(self):
        """A 1-intersection network z-scores to all-zeros; fixed scales do not."""
        scaled, _, raw = _features([_Intersection('a', 12, 12, 9, set())])
        self.assertTrue(np.any(scaled != 0.0))
        self.assertTrue(np.all(_zscore(raw) == 0.0))


def _signature(**overrides):
    signature = {
        'version': 4,
        'node_count': 16,
        'action_dim': 9,
        'phase_lengths': [9] * 16,
        'raw_state_dim': 33,
        'policy_input_dim': 33,
        'embedding_mode': 'structural',
        'meta_dim': 64,
        'structural_spec': spec_id(),
        'topology_fingerprint': None,
        'centralized_critic_mode': 'pooled',
        'graph_edge_count': 48,
    }
    signature.update(overrides)
    return signature


class ArchitectureValidationTests(unittest.TestCase):
    def test_node_count_difference_is_allowed(self):
        differing = validate_transfer_architecture(
            _signature(node_count=196, phase_lengths=[9] * 196, graph_edge_count=700),
            _signature(),
        )
        self.assertIn('node_count', differing)
        self.assertIn('phase_lengths', differing)
        self.assertIn('graph_edge_count', differing)

    def test_action_dim_difference_is_rejected_with_hint(self):
        with self.assertRaises(TransferError) as caught:
            validate_transfer_architecture(_signature(action_dim=4), _signature())
        message = str(caught.exception)
        self.assertIn('action_dim', message)
        self.assertIn('permutation-invariant phase head', message)

    def test_embedding_mode_difference_is_rejected(self):
        with self.assertRaises(TransferError) as caught:
            validate_transfer_architecture(
                _signature(), _signature(embedding_mode='learned')
            )
        self.assertIn('embedding_mode', str(caught.exception))

    def test_structural_spec_change_is_rejected(self):
        with self.assertRaises(TransferError) as caught:
            validate_transfer_architecture(
                _signature(), _signature(structural_spec='structural_v0:stale')
            )
        self.assertIn('structural_spec', str(caught.exception))

    def test_concat_critic_can_never_transfer(self):
        with self.assertRaises(TransferError) as caught:
            validate_transfer_architecture(
                _signature(centralized_critic_mode='concat'),
                _signature(centralized_critic_mode='concat'),
            )
        self.assertIn('concat', str(caught.exception))

    def test_missing_signature_is_rejected(self):
        with self.assertRaises(TransferError):
            validate_transfer_architecture(_signature(), None)


class _FakeAgent:
    """Minimal duck-typed stand-in for HyperLightPPOAgent."""

    def __init__(self, signature, actor_hypernet, agent_embeddings=None):
        self.device = torch.device('cpu')
        self._signature = signature
        self.actor_hypernet = actor_hypernet
        self.agent_embeddings = agent_embeddings

    def _architecture_signature(self):
        return self._signature


class TransferLoadTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.tmpdir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmpdir.name, 'source.pt')

    def tearDown(self):
        self.tmpdir.cleanup()

    def _save_source(self, module, signature, agent_embeddings):
        torch.save(
            {
                'architecture': signature,
                'actor_hypernet': module.state_dict(),
                'agent_embeddings': agent_embeddings,
                'optimizer': {'state': 'should never be read'},
            },
            self.path,
        )

    def test_weights_transfer_across_node_counts(self):
        source_module = nn.Linear(64, 32)
        self._save_source(
            source_module,
            _signature(node_count=16, phase_lengths=[9] * 16, graph_edge_count=48),
            torch.zeros(16, 64),
        )

        target_module = nn.Linear(64, 32)
        nn.init.constant_(target_module.weight, 0.0)
        target = _FakeAgent(
            _signature(node_count=196, phase_lengths=[9] * 196, graph_edge_count=700),
            target_module,
        )

        report = load_for_transfer(target, self.path)

        torch.testing.assert_close(target_module.weight, source_module.weight)
        self.assertEqual(report['source_node_count'], 16)
        self.assertEqual(report['target_node_count'], 196)
        self.assertEqual(report['modules']['actor_hypernet']['loaded'], 2)
        self.assertEqual(report['modules']['actor_hypernet']['skipped'], [])
        self.assertIn('node_count', report['keys_allowed_to_differ'])
        self.assertIn('16 -> 196', format_report(report))

    def test_mismatched_shapes_are_skipped_not_fatal(self):
        self._save_source(
            nn.Linear(64, 32),
            _signature(),
            torch.zeros(16, 64),
        )
        target_module = nn.Linear(64, 48)  # different output width
        before = target_module.weight.detach().clone()
        target = _FakeAgent(_signature(), target_module)

        report = load_for_transfer(target, self.path)

        torch.testing.assert_close(target_module.weight, before)
        stats = report['modules']['actor_hypernet']
        self.assertEqual(stats['loaded'], 0)
        self.assertEqual(len(stats['skipped']), 2)
        self.assertEqual(sorted(stats['missing']), ['bias', 'weight'])

    def test_strict_mode_refuses_partial_transfer(self):
        self._save_source(nn.Linear(64, 32), _signature(), torch.zeros(16, 64))
        target = _FakeAgent(_signature(), nn.Linear(64, 48))
        with self.assertRaises(TransferError):
            load_for_transfer(target, self.path, strict=True)

    def test_agent_embeddings_are_never_transferred(self):
        self._save_source(nn.Linear(64, 32), _signature(), torch.ones(16, 64))
        embeddings = nn.Parameter(torch.zeros(196, 64))
        target = _FakeAgent(_signature(node_count=196), nn.Linear(64, 32), embeddings)

        report = load_for_transfer(target, self.path)

        self.assertTrue(torch.all(embeddings == 0.0))
        self.assertTrue(
            any('agent_embeddings' in note for note in report['skipped_by_design'])
        )

    def test_missing_file_raises_transfer_error(self):
        target = _FakeAgent(_signature(), nn.Linear(64, 32))
        with self.assertRaises(TransferError):
            load_for_transfer(target, os.path.join(self.tmpdir.name, 'nope.pt'))


if __name__ == '__main__':
    unittest.main()
