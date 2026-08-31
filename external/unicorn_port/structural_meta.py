"""Network-independent structural features, rebuilt against Unicorn's Tls objects.

This is transfer/structural.py's contract expressed in the other harness. The
feature list, the order and the fixed scales are deliberately the same, so a
number produced here means what it means in our own tables. What differs is only
where the raw quantities come from: our version walks ``world.intersections``,
this one reads the ``config_data`` dict Unicorn's ``env.tls.Tls`` is built from
(maps/<net>/<net>_config.json).

Why fixed scales rather than z-scoring against the loaded network: an
intersection with 12 incoming lanes has to produce the same vector in every
roadnet, otherwise a code learned in one network means something else in the
next. Unicorn's own ``int_attr_vec`` follows the same rule (it divides lengths
by 300, speeds by 30, lane counts by 8), which is worth knowing before reading
any result from this port -- see README.md.
"""

import numpy as np

# (name, fixed scale). Identical to transfer/structural.py's _FEATURE_SPEC.
FEATURE_SPEC = (
    ('in_lane_count', 20.0),
    ('out_lane_count', 20.0),
    ('in_degree', 8.0),
    ('out_degree', 8.0),
    ('node_degree', 16.0),
    ('neighbor_count', 8.0),
    ('phase_count', 12.0),
    ('startlane_count', 20.0),
    ('lanes_per_in_road', 8.0),
    ('lanes_per_out_road', 8.0),
    ('out_in_lane_ratio', 4.0),
    ('controlled_neighbor_ratio', 1.0),
)

FEATURE_NAMES = tuple(name for name, _ in FEATURE_SPEC)
FEATURE_DIM = len(FEATURE_SPEC)
_SCALES = np.asarray([scale for _, scale in FEATURE_SPEC], dtype=np.float32)

SPEC_ID = 'structural_v1_unicorn:' + ','.join(FEATURE_NAMES)


def _edge_count(lane_list):
    """Approaches, not lanes: SUMO lane ids are '<edge>_<index>'.

    Unicorn's config carries incoming_lane_list but not always an edge list, and
    deriving the edge from the lane id is what our _road_key does on the SUMO
    side too, so the two harnesses agree on what "degree" counts.
    """
    edges = set()
    for lane in lane_list or ():
        lane = str(lane)
        edges.add(lane.rsplit('_', 1)[0] if '_' in lane else lane)
    return len(edges)


def raw_features_for(tls):
    """The twelve physical quantities for one Unicorn ``Tls``, unscaled."""
    in_lanes = list(getattr(tls, 'incoming_lane_list', None) or ())
    out_lanes = list(getattr(tls, 'outgoing_lane_list', None) or ())
    neighbours = list(getattr(tls, 'neighbor_list', None) or ())
    phases = list(getattr(tls, 'action_space', None) or ())

    in_lane_count = float(len(in_lanes))
    out_lane_count = float(len(out_lanes))
    in_degree = float(_edge_count(in_lanes))
    out_degree = float(_edge_count(out_lanes))
    neighbor_count = float(len(neighbours))
    # max(1, ...) mirrors our version: a signal with no programme still has one
    # thing it can do, and a zero here would divide the scaled feature away.
    phase_count = float(max(1, len(phases)))

    return np.asarray([
        in_lane_count,
        out_lane_count,
        in_degree,
        out_degree,
        in_degree + out_degree,
        neighbor_count,
        phase_count,
        # startlane_count is the incoming lanes an agent actually starts from,
        # which on both sides equals in_lane_count for a signalised node.
        in_lane_count,
        in_lane_count / max(1.0, in_degree),
        out_lane_count / max(1.0, out_degree),
        out_lane_count / max(1.0, in_lane_count),
        neighbor_count / max(1.0, in_degree),
    ], dtype=np.float32)


def build_meta_table(env, shrink=1.0):
    """``[n_intersections, FEATURE_DIM]`` in ``env.tls_list`` order.

    The row order is the agent order the model indexes with, so it must stay
    ``env.tls_list``; Unicorn builds its observation batches in that same order.

    ``shrink`` pulls every column toward this network's own mean,
    ``x -> mean + shrink*(x - mean)``, the knob from --structural_shrink. 1.0 is
    skipped rather than applied, because ``mean + 1.0*(x - mean)`` is not
    bit-for-bit ``x`` in float32.
    """
    rows = [raw_features_for(env.tls_dict[tls]) for tls in env.tls_list]
    raw = np.stack(rows, axis=0)
    scaled = (raw / _SCALES).astype(np.float32)
    if shrink is not None and float(shrink) != 1.0:
        factor = float(shrink)
        if factor < 0.0:
            raise ValueError('structural shrink must be >= 0, got %r' % (shrink,))
        column_mean = scaled.mean(axis=0, keepdims=True)
        scaled = (column_mean + factor * (scaled - column_mean)).astype(np.float32)
    return scaled, raw


def summarize(raw):
    """One log line of min/mean/max per feature, same format as our BRF logs."""
    parts = []
    for i, name in enumerate(FEATURE_NAMES):
        col = raw[:, i]
        parts.append('%s[%g/%.2f/%g]' % (name, col.min(), col.mean(), col.max()))
    return 'structural features (min/mean/max): ' + ' '.join(parts)
