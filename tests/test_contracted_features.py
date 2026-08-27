"""The opt-in contracted-neighbourhood features.

Why they exist: on the CityFlow grids every feature in the v1 contract is
constant across intersections except neighbor_count, and on Ingolstadt21
neighbor_count is zero at 19 of 21 -- so no entry of the contract varies
usefully on both. Counting neighbours through the contracted adjacency
reproduces neighbor_count exactly on the grids and spreads 0-13 on
Ingolstadt21.

These must be asked for by name. The default contract is untouched, which is
what keeps every checkpoint written before them loadable.
"""

import unittest

import numpy as np

from transfer import (
    ALL_FEATURE_NAMES,
    EXTENDED_FEATURE_NAMES,
    FEATURE_DIM,
    FEATURE_NAMES,
    build_structural_features,
    resolve_features,
    spec_id,
)


def _road(lanes, road_id):
    return {'id': road_id, 'lanes': list(range(lanes))}


class _Inter:
    def __init__(self, inter_id, phases, in_ids, out_ids, lanes=3):
        self.id = inter_id
        self.in_roads = [_road(lanes, r) for r in in_ids]
        self.out_roads = [_road(lanes, r) for r in out_ids]
        self.phases = list(range(phases))
        self.startlanes = list(range(lanes * len(in_ids)))


def _net():
    return [
        _Inter('a', 4, ['r_b_a', 'r_x_a'], ['r_a_b']),
        _Inter('b', 4, ['r_a_b'], ['r_b_a']),
        _Inter('c', 2, ['r_y_c'], ['r_c_y']),
    ]


def _build(features, degrees=None):
    return build_structural_features(
        _net(),
        lanes_for_road=lambda inter, road: list(road.get('lanes', [])),
        features=features,
        contracted_degrees=degrees,
    )


SWAPPED = ','.join(
    [n for n in FEATURE_NAMES if n not in ('neighbor_count', 'controlled_neighbor_ratio')]
    + list(EXTENDED_FEATURE_NAMES)
)


class ExtendedFeatureTests(unittest.TestCase):
    def test_default_contract_excludes_them(self):
        self.assertEqual(len(FEATURE_NAMES), FEATURE_DIM)
        for name in EXTENDED_FEATURE_NAMES:
            self.assertNotIn(name, FEATURE_NAMES)
            self.assertIn(name, ALL_FEATURE_NAMES)
        self.assertNotIn('contracted', spec_id())

    def test_default_spec_is_byte_identical(self):
        self.assertEqual(spec_id(), 'structural_v1:' + ','.join(FEATURE_NAMES))

    def test_extended_spec_is_flagged(self):
        self.assertTrue(spec_id(SWAPPED).startswith('structural_v1+ext:'))
        self.assertNotEqual(spec_id(SWAPPED), spec_id())

    def test_values_come_from_the_supplied_degrees(self):
        scaled, names, raw = _build(SWAPPED, degrees=[5.0, 1.0, 0.0])
        count = raw[:, names.index('contracted_neighbor_count')]
        np.testing.assert_array_equal(count, np.array([5.0, 1.0, 0.0]))

    def test_ratio_divides_by_in_degree(self):
        _, names, raw = _build(SWAPPED, degrees=[5.0, 1.0, 0.0])
        ratio = raw[:, names.index('contracted_neighbor_ratio')]
        # in_degree is 2, 1, 1 for a, b, c
        np.testing.assert_allclose(ratio, np.array([2.5, 1.0, 0.0]))

    def test_contract_columns_are_untouched_by_the_swap(self):
        _, base_names, base_raw = _build(None)
        _, names, raw = _build(SWAPPED, degrees=[5.0, 1.0, 0.0])
        for name in names:
            if name in EXTENDED_FEATURE_NAMES:
                continue
            np.testing.assert_array_equal(
                raw[:, names.index(name)], base_raw[:, base_names.index(name)])

    def test_missing_degrees_is_refused(self):
        with self.assertRaises(ValueError):
            _build(SWAPPED)

    def test_wrong_length_degrees_is_refused(self):
        with self.assertRaises(ValueError):
            _build(SWAPPED, degrees=[1.0, 2.0])

    def test_degrees_are_ignored_when_not_requested(self):
        # the v1 path must not change shape or content just because degrees exist
        _, names, raw = _build(None, degrees=[5.0, 1.0, 0.0])
        self.assertEqual(list(names), list(FEATURE_NAMES))
        self.assertEqual(raw.shape[1], FEATURE_DIM)


if __name__ == '__main__':
    unittest.main()
