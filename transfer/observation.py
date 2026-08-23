"""Capacity-normalised observations.

Why this exists
---------------
``get_ob`` divides every per-lane count by a single global constant
(``vehicle_max``, 50 by default). That makes the observation incomparable
between lanes of different length, both across networks and within one:

    network            road lengths
    hangzhou 4x4       600 m and 800 m
    manhattan 16x3     100 m and 350 m
    manhattan 28x7     300 m throughout

Ten vehicles on a 600 m lane is an almost empty approach; ten vehicles on a
100 m lane is a nearly full one. Under the fixed constant both arrive at the
policy as 0.2. A policy trained on 4x4 therefore reads "0.2" as free flow and
carries that reading onto 16x3, where the same number means jammed.

The BRSC-MAPPO study measured exactly this: ablating its ``incoming_occupancy``
dimension -- vehicles divided by the lane's own geometric capacity -- cost
+29.3 s of transferred travel time (t = 7.15), nearly the whole portability
budget. This module ports that normalisation.

Capacity follows the same definition: lane length divided by an effective
per-vehicle headway (7.5 m), and the resulting ratio is clipped, so a
temporarily over-saturated lane cannot produce an unbounded input.
"""

from __future__ import annotations

import numpy as np

#: Metres of lane occupied by one stationary vehicle, including the gap.
DEFAULT_HEADWAY_M = 7.5

#: Occupancy is clipped here; >1 is physically possible for a moment when
#: vehicles are packed tighter than the nominal headway.
DEFAULT_CLIP = 1.5


def lane_length(world, lane_id):
    """Length of ``lane_id`` in metres, in either world, or None.

    CityFlow precomputes ``world.lane_length``; the SUMO world does not keep a
    table but its engine can be asked directly.
    """
    table = getattr(world, 'lane_length', None)
    if isinstance(table, dict):
        value = table.get(lane_id)
        if value:
            return float(value)

    engine = getattr(world, 'eng', None)
    lane_api = getattr(engine, 'lane', None)
    if lane_api is not None and hasattr(lane_api, 'getLength'):
        try:
            return float(lane_api.getLength(lane_id))
        except Exception:
            return None
    return None


def lane_capacity(world, lane_id, headway=DEFAULT_HEADWAY_M, fallback=None):
    """Vehicles a lane can hold. ``fallback`` is used when length is unknown."""
    length = lane_length(world, lane_id)
    if length is None or length <= 0.0:
        return fallback
    return max(1.0, length / float(headway))


def build_divisors(world, lane_ids, ob_length, feature_count, *,
                   headway=DEFAULT_HEADWAY_M, fallback=1.0):
    """Per-element divisor vector matching one intersection's observation.

    ``LaneVehicleGenerator`` with ``average=None`` emits one block per feature,
    each block running over ``lane_ids`` in order, and the agent then zero-pads
    the vector out to ``ob_length``. The divisors have to follow that exact
    layout, so the same lane's capacity lines up with every feature block.

    Padding positions get 1.0: those entries are zero anyway, and a zero
    divisor would produce NaN.

    Returns ``(divisors, resolved, missing)`` -- the vector, and how many lane
    lengths were found versus fell back, so the caller can log it rather than
    silently normalising by a made-up number.
    """
    divisors = np.ones((ob_length,), dtype=np.float32)
    resolved = 0
    missing = 0
    lane_count = len(lane_ids)
    for lane_index, lane_id in enumerate(lane_ids):
        capacity = lane_capacity(world, lane_id, headway=headway, fallback=None)
        if capacity is None:
            capacity = float(fallback)
            missing += 1
        else:
            resolved += 1
        for feature_index in range(feature_count):
            flat = feature_index * lane_count + lane_index
            if flat < ob_length:
                divisors[flat] = capacity
    return divisors, resolved, missing


def summarize(divisors_per_node):
    """One-line capacity summary for the run log."""
    if not divisors_per_node:
        return 'lane capacity: <none>'
    values = np.concatenate([
        np.asarray(d, dtype=np.float32).reshape(-1) for d in divisors_per_node
    ])
    real = values[values > 1.0]
    if real.size == 0:
        return 'lane capacity: no lane length resolved'
    return (
        f'lane capacity (vehicles/lane): min={real.min():.1f} '
        f'mean={real.mean():.1f} max={real.max():.1f}'
    )
