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

_FEATURE_INDEX = {name: idx for idx, name in enumerate(FEATURE_NAMES)}


def resolve_features(features=None):
    """Normalise a feature selection into ``(names, indices)``.

    ``features`` is ``None`` (the full contract), a comma-separated string, or a
    sequence of names.  The result is always in the canonical ``_FEATURE_SPEC``
    order regardless of the order the caller wrote them in, so the same subset
    can only ever produce one ``spec_id``.
    """
    if features is None:
        return FEATURE_NAMES, np.arange(FEATURE_DIM)
    if isinstance(features, str):
        wanted = [part.strip() for part in features.split(',') if part.strip()]
    else:
        wanted = [str(part).strip() for part in features if str(part).strip()]
    if not wanted:
        raise ValueError('structural feature selection is empty')
    unknown = [name for name in wanted if name not in _FEATURE_INDEX]
    if unknown:
        raise ValueError(
            'unknown structural feature(s): ' + ', '.join(unknown)
            + '; valid names are ' + ', '.join(FEATURE_NAMES)
        )
    seen = set()
    duplicates = [n for n in wanted if n in seen or seen.add(n)]
    if duplicates:
        raise ValueError('duplicate structural feature(s): ' + ', '.join(sorted(set(duplicates))))
    indices = np.asarray(sorted(_FEATURE_INDEX[name] for name in wanted))
    return tuple(FEATURE_NAMES[i] for i in indices), indices


def spec_id(features=None):
    """Stable identifier for the feature contract, stored in checkpoints.

    A subset produces a different id from the full contract, which is what stops
    a 4-feature run from loading a 12-feature checkpoint (and vice versa).
    """
    names, _ = resolve_features(features)
    return f"structural_v{SPEC_VERSION}:" + ','.join(names)


def _road_key(road):
    """Identity of a road, in either world.

    CityFlow hands the agent roadnet dicts; the SUMO world hands it plain road-id
    strings. Both need a hashable key so shared roads can be matched.
    """
    if isinstance(road, dict):
        return road.get('id')
    return road


def _neighbour_sets(intersections):
    """Map each intersection to the other *controlled* ones it shares a road with.

    Deriving this from shared roads rather than from a roadnet dict field is what
    makes it work on both worlds: an intersection j is a neighbour of i when some
    road leaves j and enters i (or the reverse).
    """
    in_keys = []
    out_keys = []
    producers = {}
    consumers = {}
    for idx, inter in enumerate(intersections):
        ins = {_road_key(r) for r in (getattr(inter, 'in_roads', []) or [])}
        outs = {_road_key(r) for r in (getattr(inter, 'out_roads', []) or [])}
        ins.discard(None)
        outs.discard(None)
        in_keys.append(ins)
        out_keys.append(outs)
        for key in outs:
            producers.setdefault(key, set()).add(idx)
        for key in ins:
            consumers.setdefault(key, set()).add(idx)

    neighbours = []
    for idx in range(len(intersections)):
        found = set()
        for key in in_keys[idx]:
            found |= producers.get(key, set())
        for key in out_keys[idx]:
            found |= consumers.get(key, set())
        found.discard(idx)
        neighbours.append(found)
    return neighbours


def build_structural_features(intersections, *, lanes_for_road, features=None):
    """Build ``[n_intersections, FEATURE_DIM]`` network-independent features.

    ``lanes_for_road(inter, road)`` is passed in as a callable -- the agent's
    ``_lanes_for_road``, which already resolves a road to its lane ids in both
    the CityFlow and SUMO worlds -- so this module stays free of any simulator
    imports and can be unit tested with plain stubs.

    Returns ``(scaled, names, raw)``.  ``raw`` is kept for logging: it is the
    unscaled physical quantity, which is what a human wants to read in a log.
    """
    neighbours = _neighbour_sets(intersections)
    rows = []
    for idx, inter in enumerate(intersections):
        in_roads = getattr(inter, 'in_roads', []) or []
        out_roads = getattr(inter, 'out_roads', []) or []

        in_lane_count = float(sum(len(lanes_for_road(inter, road)) for road in in_roads))
        out_lane_count = float(sum(len(lanes_for_road(inter, road)) for road in out_roads))
        in_degree = float(len(in_roads))
        out_degree = float(len(out_roads))
        node_degree = in_degree + out_degree
        neighbor_count = float(len(neighbours[idx]))
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
    scaled = (raw / _SCALES).astype(np.float32)
    names, indices = resolve_features(features)
    if len(indices) != FEATURE_DIM:
        scaled = scaled[:, indices]
        raw = raw[:, indices]
    return scaled, list(names), raw


def summarize_raw_features(raw, names=None):
    """One-line-per-feature min/mean/max summary for the run log."""
    if raw is None or len(raw) == 0:
        return 'structural features: <empty>'
    array = np.asarray(raw, dtype=np.float32)
    if names is None:
        names = FEATURE_NAMES if array.shape[1] == FEATURE_DIM else             tuple(f'f{i}' for i in range(array.shape[1]))
    parts = []
    for idx, name in enumerate(names):
        column = array[:, idx]
        parts.append(f'{name}[{column.min():g}/{column.mean():.2f}/{column.max():g}]')
    return 'structural features (min/mean/max): ' + ' '.join(parts)
