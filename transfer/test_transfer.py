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
    resolve_features,
    format_report,
    load_for_transfer,
    spec_id,
    validate_transfer_architecture,
)


def _road(lanes, road_id):
    return {'id': road_id, 'lanes': list(range(lanes))}


class _Intersection:
    """Stub intersection carrying only what the feature builder reads.

    Roads are identified the way CityFlow does it (dicts with an id); the SUMO
    path, where roads are plain id strings resolved through
    ``inter.road_lane_mapping``, is covered by the agent's ``_lanes_for_road``
    rather than here.
    """

    def __init__(self, inter_id, phases, in_road_ids, out_road_ids, lanes_per_road=3):
        self.id = inter_id
        self.in_roads = [_road(lanes_per_road, rid) for rid in in_road_ids]
        self.out_roads = [_road(lanes_per_road, rid) for rid in out_road_ids]
        self.phases = list(range(phases))
        self.startlanes = list(range(lanes_per_road * len(in_road_ids)))


def _features(intersections):
    return build_structural_features(
        intersections,
        lanes_for_road=lambda inter, road: list(road.get('lanes', [])),
    )


def _network(filler_count, probe_neighbours=2):
    """A 4-way 'probe' intersection embedded in a network of a chosen size.

    ``probe`` always has the same shape and the same number of *controlled*
    neighbours; only the surrounding population changes. Filler intersections
    deliberately vary in size so the per-network mean/std move.
    """
    in_ids, out_ids = [], []
    for i in range(4):
        if i < probe_neighbours:
            in_ids.append(f'r_n{i}_probe')
            out_ids.append(f'r_probe_n{i}')
        else:
            in_ids.append(f'r_ext{i}_probe')
            out_ids.append(f'r_probe_ext{i}')

    inters = [_Intersection('probe', 8, in_ids, out_ids)]
    for i in range(probe_neighbours):
        inters.append(
            _Intersection(f'n{i}', 8, [f'r_probe_n{i}'], [f'r_n{i}_probe'])
        )
    for j in range(filler_count):
        inters.append(
            _Intersection(
                f'f{j}',
                2 + (j % 6),
                [f'r_f{j}_in'],
                [f'r_f{j}_out'],
                lanes_per_road=1 + (j % 4),
            )
        )
    return inters


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

        A given intersection must map to the same conditioning vector no matter
        what the rest of the network looks like.
        """
        small_scaled, _, small_raw = _features(_network(2))
        large_scaled, _, large_raw = _features(_network(40))

        np.testing.assert_allclose(small_scaled[0], large_scaled[0], rtol=0, atol=0)

        # ...and the legacy per-network z-score does NOT have that property,
        # which is exactly why this module exists.
        self.assertFalse(
            np.allclose(_zscore(small_raw)[0], _zscore(large_raw)[0]),
            'legacy z-score unexpectedly stable; the regression this guards is gone',
        )

    def test_neighbours_come_from_shared_roads(self):
        """Adjacency is derived, not declared, so it works in both worlds."""
        _, _, raw = _features(_network(5, probe_neighbours=3))
        neighbour_idx = FEATURE_NAMES.index('neighbor_count')
        ratio_idx = FEATURE_NAMES.index('controlled_neighbor_ratio')
        self.assertEqual(raw[0, neighbour_idx], 3.0)
        # 3 of the probe's 4 approaches lead to a controlled intersection
        self.assertAlmostEqual(float(raw[0, ratio_idx]), 0.75, places=6)

        _, _, isolated = _features(_network(5, probe_neighbours=0))
        self.assertEqual(isolated[0, neighbour_idx], 0.0)

    def test_features_are_finite_and_scaled(self):
        scaled, names, raw = _features(_network(3))
        self.assertEqual(scaled.shape, (6, FEATURE_DIM))
        self.assertEqual(names, list(FEATURE_NAMES))
        self.assertTrue(np.isfinite(scaled).all())
        self.assertTrue((np.abs(scaled) <= 3.0).all(), scaled)
        # raw keeps the physical quantity for logging: 4 approaches x 3 lanes
        self.assertEqual(raw[0, FEATURE_NAMES.index('in_lane_count')], 12.0)

    def test_single_intersection_network_does_not_collapse(self):
        """A 1-intersection network z-scores to all-zeros; fixed scales do not."""
        lone = [_Intersection('a', 8, ['r_x_a'], ['r_a_x'])]
        scaled, _, raw = _features(lone)
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


class StructuralFeatureSubsetTests(unittest.TestCase):
    """A subset must drop columns without disturbing the ones it keeps."""

    SUBSET = 'in_lane_count,out_lane_count,in_degree,out_degree'

    def test_default_spec_is_unchanged(self):
        # Existing checkpoints carry this exact string; changing it would make
        # every stored architecture signature unloadable.
        self.assertEqual(spec_id(), spec_id(None))
        self.assertTrue(spec_id().endswith(',controlled_neighbor_ratio'))
        self.assertEqual(spec_id().count(','), FEATURE_DIM - 1)

    def test_subset_spec_differs_from_full(self):
        self.assertNotEqual(spec_id(self.SUBSET), spec_id())
        self.assertEqual(
            spec_id(self.SUBSET),
            'structural_v1:in_lane_count,out_lane_count,in_degree,out_degree',
        )

    def test_selection_is_canonically_ordered(self):
        shuffled = 'out_degree,in_lane_count,out_lane_count,in_degree'
        self.assertEqual(spec_id(shuffled), spec_id(self.SUBSET))
        names, indices = resolve_features(shuffled)
        self.assertEqual(list(names), self.SUBSET.split(','))
        self.assertEqual(list(indices), [0, 1, 2, 3])

    def test_subset_columns_match_the_full_build(self):
        intersections = _network(3)
        full_scaled, full_names, full_raw = _features(intersections)
        sub_scaled, sub_names, sub_raw = build_structural_features(
            intersections,
            lanes_for_road=lambda inter, road: list(road.get('lanes', [])),
            features=self.SUBSET,
        )
        self.assertEqual(sub_names, self.SUBSET.split(','))
        self.assertEqual(sub_scaled.shape, (len(intersections), 4))
        for name in sub_names:
            col = full_names.index(name)
            np.testing.assert_array_equal(
                sub_scaled[:, sub_names.index(name)], full_scaled[:, col]
            )
            np.testing.assert_array_equal(
                sub_raw[:, sub_names.index(name)], full_raw[:, col]
            )

    def test_unknown_duplicate_and_empty_are_rejected(self):
        for bad in ('nope', 'in_degree,in_degree', '', '  ,  '):
            with self.assertRaises(ValueError):
                resolve_features(bad)


if __name__ == '__main__':
    unittest.main()
