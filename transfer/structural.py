"""Network-independent structural conditioning features.

Why this module exists
----------------------
``HyperLightPPOAgent._build_topology_features`` already extracts the right raw
quantities (lane counts, degrees, phase counts...), but it finishes with
``_normalize_topology_features``, which z-scores every column using the mean and
std **of the network currently loaded**.  That is fine for a single-network run
and fatal for transfer: a "12 incoming lanes, 9 phases" intersection maps to one
meta vector inside 4x4 and to a completely different one inside Ingolstadt21,
because the population it is being standardised against changed.  A hypernetwork
trained on the first mapping cannot be reused under the second.

This module produces the same kind of features but scales them by **fixed
constants that never depend on the loaded network**, so the same physical
intersection always yields the same vector.  It also drops the absolute ``x``/
``y`` coordinates, which carry no transferable meaning (every roadnet has its
own origin and units), and adds a few scale-free ratios that describe an
intersection's role rather than its size.

The feature list, its order, and the constants are all part of the *contract*
between a source checkpoint and a target run.  ``spec_id()`` fingerprints that
contract and is written into the checkpoint architecture signature, so a run
that silently changes the spec cannot load an older checkpoint.
"""

from __future__ import annotations

import numpy as np

#: Bump whenever a feature is added/removed/reordered or a scale changes.
SPEC_VERSION = 1

# (name, fixed scale).  The scale is a plain constant chosen to put typical
# values in roughly [0, 1.5]; it is NEVER derived from the loaded roadnet.
# Rough anchors: a big urban intersection has ~20 incoming lanes, 8 approach
# roads, and up to ~12 signal phases.
_FEATURE_SPEC = (
    ('in_lane_count', 20.0),
    ('out_lane_count', 20.0),
    ('in_degree', 8.0),
    ('out_degree', 8.0),
    ('node_degree', 16.0),
    ('neighbor_count', 8.0),
    ('phase_count', 12.0),
    ('startlane_count', 20.0),
    # Scale-free ratios: shape/role of the intersection, not its size.
    ('lanes_per_in_road', 8.0),
    ('lanes_per_out_road', 8.0),
    ('out_in_lane_ratio', 4.0),
    ('controlled_neighbor_ratio', 1.0),
)

FEATURE_NAMES = tuple(name for name, _ in _FEATURE_SPEC)
FEATURE_DIM = len(_FEATURE_SPEC)
_SCALES = np.asarray([scale for _, scale in _FEATURE_SPEC], dtype=np.float32)


def spec_id():
    """Stable identifier for the feature contract, stored in checkpoints."""
    return f"structural_v{SPEC_VERSION}:" + ','.join(FEATURE_NAMES)


def build_structural_features(intersections, *, road_lane_count, neighbor_intersections):
    """Build ``[n_intersections, FEATURE_DIM]`` network-independent features.

    ``road_lane_count`` and ``neighbor_intersections`` are passed in as callables
    (the agent already owns those helpers) so this module stays free of any
    simulator/world imports and can be unit tested with plain stubs.

    Returns ``(scaled, names, raw)``.  ``raw`` is kept for logging: it is the
    unscaled physical quantity, which is what a human wants to read in a log.
    """
    rows = []
    for inter in intersections:
        in_roads = getattr(inter, 'in_roads', []) or []
        out_roads = getattr(inter, 'out_roads', []) or []

        in_lane_count = float(sum(road_lane_count(road) for road in in_roads))
        out_lane_count = float(sum(road_lane_count(road) for road in out_roads))
        in_degree = float(len(in_roads))
        out_degree = float(len(out_roads))
        node_degree = in_degree + out_degree
        neighbor_count = float(len(neighbor_intersections(inter)))
        phase_count = float(max(1, len(getattr(inter, 'phases', []) or [])))
        startlane_count = float(len(getattr(inter, 'startlanes', []) or []))

        rows.append(
            [
                in_lane_count,
                out_lane_count,
                in_degree,
                out_degree,
                node_degree,
                neighbor_count,
                phase_count,
                startlane_count,
                in_lane_count / max(1.0, in_degree),
                out_lane_count / max(1.0, out_degree),
                out_lane_count / max(1.0, in_lane_count),
                # How many approaches lead to another *controlled* intersection.
                # 1.0 deep inside the network, lower at the boundary -- the one
                # feature that tells the policy it is sitting on the edge of the
                # controlled area, and it stays meaningful in any roadnet.
                neighbor_count / max(1.0, in_degree),
            ]
        )

    raw = np.asarray(rows, dtype=np.float32).reshape(len(rows), FEATURE_DIM)
    return (raw / _SCALES).astype(np.float32), list(FEATURE_NAMES), raw


def summarize_raw_features(raw):
    """One-line-per-feature min/mean/max summary for the run log."""
    if raw is None or len(raw) == 0:
        return 'structural features: <empty>'
    array = np.asarray(raw, dtype=np.float32)
    parts = []
    for idx, name in enumerate(FEATURE_NAMES):
        column = array[:, idx]
        parts.append(f'{name}[{column.min():g}/{column.mean():.2f}/{column.max():g}]')
    return 'structural features (min/mean/max): ' + ' '.join(parts)
