"""Unit tests for capacity-normalised observations.

Run inside the container:

    cd /DaRL/LibSignal && python -m unittest transfer.test_observation -v
"""

import unittest
from types import SimpleNamespace

import numpy as np

from transfer.observation import (
    DEFAULT_CLIP,
    DEFAULT_HEADWAY_M,
    build_divisors,
    lane_capacity,
    lane_length,
    summarize,
)


class _CityFlowWorld:
    def __init__(self, lengths):
        self.lane_length = dict(lengths)


class _SumoWorld:
    """Engine-only: no table, the engine answers per lane.

    world_sumo now builds a ``lane_length`` table of its own, so this is the
    fallback path rather than the normal one.  It is kept because the fallback
    has to keep working for any world that does not build a table.
    """

    def __init__(self, lengths):
        self._lengths = dict(lengths)
        self.eng = SimpleNamespace(lane=SimpleNamespace(getLength=self._get))

    def _get(self, lane_id):
        if lane_id not in self._lengths:
            raise KeyError(lane_id)
        return self._lengths[lane_id]


class _SumoWorldClosedEngine:
    """A SUMO world after ``World.__init__`` has closed its connection.

    This is the state the agent is constructed in: ``libsumo.close()`` has run,
    so every lane query raises "A network was not yet constructed".  The table
    read while the engine was alive is the only source left, and it is why
    world_sumo builds one.
    """

    def __init__(self, lengths):
        self.lane_length = dict(lengths)
        self.eng = SimpleNamespace(lane=SimpleNamespace(getLength=self._dead))

    @staticmethod
    def _dead(lane_id):
        raise RuntimeError('A network was not yet constructed.')


class LaneLengthTests(unittest.TestCase):
    def test_reads_the_cityflow_table(self):
        world = _CityFlowWorld({'a_0': 600.0})
        self.assertEqual(lane_length(world, 'a_0'), 600.0)

    def test_falls_through_to_the_sumo_engine(self):
        world = _SumoWorld({'a_0': 123.0})
        self.assertEqual(lane_length(world, 'a_0'), 123.0)

    def test_unknown_lane_is_none_not_an_exception(self):
        self.assertIsNone(lane_length(_CityFlowWorld({}), 'nope'))
        self.assertIsNone(lane_length(_SumoWorld({}), 'nope'))

    def test_a_closed_sumo_engine_still_resolves_from_the_table(self):
        # The agent is built after World.__init__ closes the connection, so
        # without the table this returns None for every lane and capacity
        # normalisation silently degrades to the vehicle_max fallback.
        world = _SumoWorldClosedEngine({'a_0': 82.5})
        self.assertEqual(lane_length(world, 'a_0'), 82.5)
        self.assertAlmostEqual(lane_capacity(world, 'a_0'), 82.5 / DEFAULT_HEADWAY_M)

    def test_a_closed_sumo_engine_without_a_table_reports_unknown(self):
        world = _SumoWorldClosedEngine({})
        self.assertIsNone(lane_length(world, 'a_0'))

    def test_capacity_is_length_over_headway(self):
        world = _CityFlowWorld({'a_0': 600.0})
        self.assertAlmostEqual(lane_capacity(world, 'a_0'), 600.0 / DEFAULT_HEADWAY_M)

    def test_capacity_uses_the_fallback_when_length_is_unknown(self):
        world = _CityFlowWorld({})
        self.assertEqual(lane_capacity(world, 'nope', fallback=50.0), 50.0)

    def test_capacity_never_drops_below_one_vehicle(self):
        world = _CityFlowWorld({'tiny': 1.0})
        self.assertEqual(lane_capacity(world, 'tiny'), 1.0)


class DivisorLayoutTests(unittest.TestCase):
    def test_every_feature_block_reuses_the_same_lane_capacity(self):
        """The generator emits one block per feature over the same lanes, so a
        lane's capacity has to appear once in each block at the same offset."""
        world = _CityFlowWorld({'a_0': 75.0, 'a_1': 150.0})
        divisors, resolved, missing = build_divisors(
            world, ['a_0', 'a_1'], ob_length=4, feature_count=2
        )
        np.testing.assert_allclose(divisors, [10.0, 20.0, 10.0, 20.0])
        self.assertEqual((resolved, missing), (2, 0))

    def test_padding_positions_stay_at_one(self):
        """Padding entries are zero; a zero divisor would make them NaN."""
        world = _CityFlowWorld({'a_0': 75.0})
        divisors, _, _ = build_divisors(
            world, ['a_0'], ob_length=4, feature_count=2
        )
        np.testing.assert_allclose(divisors, [10.0, 10.0, 1.0, 1.0])
        self.assertTrue(np.all(divisors > 0.0))

    def test_unresolvable_lanes_are_counted_not_hidden(self):
        world = _CityFlowWorld({'a_0': 75.0})
        divisors, resolved, missing = build_divisors(
            world, ['a_0', 'ghost'], ob_length=2, feature_count=1, fallback=50.0
        )
        np.testing.assert_allclose(divisors, [10.0, 50.0])
        self.assertEqual((resolved, missing), (1, 1))

    def test_a_lane_that_overflows_ob_length_is_dropped_safely(self):
        world = _CityFlowWorld({'a_0': 75.0, 'a_1': 150.0})
        divisors, _, _ = build_divisors(
            world, ['a_0', 'a_1'], ob_length=1, feature_count=1
        )
        self.assertEqual(divisors.shape, (1,))
        self.assertAlmostEqual(float(divisors[0]), 10.0)


class NormalisationBehaviourTests(unittest.TestCase):
    def test_same_count_maps_to_different_occupancy_on_different_lanes(self):
        """The whole point: ten vehicles is empty on a 600 m lane and nearly
        full on a 100 m one, and the fixed constant hides that."""
        world = _CityFlowWorld({'long': 600.0, 'short': 100.0})
        divisors, _, _ = build_divisors(
            world, ['long', 'short'], ob_length=2, feature_count=1
        )
        counts = np.asarray([10.0, 10.0], dtype=np.float32)

        capacity_view = np.clip(counts / divisors, 0.0, DEFAULT_CLIP)
        fixed_view = counts / 50.0

        self.assertAlmostEqual(float(capacity_view[0]), 0.125, places=3)
        self.assertAlmostEqual(float(capacity_view[1]), 0.75, places=3)
        np.testing.assert_allclose(fixed_view, [0.2, 0.2])

    def test_clip_bounds_an_oversaturated_lane(self):
        world = _CityFlowWorld({'short': 75.0})  # capacity 10
        divisors, _, _ = build_divisors(
            world, ['short'], ob_length=1, feature_count=1
        )
        packed = np.asarray([40.0], dtype=np.float32)
        self.assertEqual(
            float(np.clip(packed / divisors, 0.0, DEFAULT_CLIP)[0]),
            DEFAULT_CLIP,
        )

    def test_summarize_reports_real_capacities_only(self):
        world = _CityFlowWorld({'a_0': 750.0})
        divisors, _, _ = build_divisors(
            world, ['a_0'], ob_length=3, feature_count=1
        )
        text = summarize([divisors])
        self.assertIn('100.0', text)  # 750 / 7.5, padding 1.0 excluded

    def test_summarize_says_so_when_nothing_resolved(self):
        self.assertIn('no lane length', summarize([np.ones(4, dtype=np.float32)]))


if __name__ == '__main__':
    unittest.main()
