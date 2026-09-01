"""Kill test for the relational (edge-conditioned) hypernetwork.

Proposal A generates a message function per *edge* from that edge's geometry.
It is worth building only if two things hold on the networks this study uses:

  1. the contracted signal-to-signal graph is not almost empty -- CoLight was
     found lazy on Ingolstadt21 because the road-adjacency there is 21 nodes
     and 2 edges, so 19 signals only ever attended to themselves; and
  2. the edge geometry actually *varies*, because the whole reason to condition
     on edges rather than nodes is that node features are constant on the grids
     (12 in-lane, 8 phases everywhere) while edges need not be.

Pass condition, fixed before running: mean in-degree >= 2 and CV(tau) >= 0.2.

This reads roadnet files only -- no simulator, no world, no torch -- so it is
cheap enough to run over every network before any training is launched.
Adjacency is built with the same contraction the rest of the repo uses
(``common.utils.contract_uncontrolled``), and the edge set produced here is
asserted against it, so this cannot silently measure a different graph.
"""

import argparse
import json
import math
import os
import sys
from heapq import heappop, heappush

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.utils import (  # noqa: E402
    build_index_intersection_map_cityflow,
    build_index_intersection_map_sumo,
    contract_uncontrolled,
)

# (network, world) pairs this study reports on. cologne3 and atlanta_1x5 are
# the heterogeneous transfer targets; 4x4_hetero is the CityFlow-side control
# for the "heterogeneity == SUMO" confound.
NETWORKS = [
    ('cityflow4x4', 'cityflow'),
    ('cityflow4x4_hetero', 'cityflow'),
    ('cityflow16x3', 'cityflow'),
    ('cityflow7x28', 'cityflow'),
    ('sumo1x21', 'sumo'),
    ('sumo_cologne3', 'sumo'),
    ('sumo_atlanta1x5', 'sumo'),
]

DEFAULT_SPEED = 13.89  # m/s, only used when a roadnet omits a speed limit


def roadnet_path(network):
    with open(os.path.join('configs', 'sim', network + '.cfg')) as handle:
        cfg = json.load(handle)
    return os.path.join(cfg['dir'], cfg['roadnetFile'])


def _polyline_length(points):
    total = 0.0
    for a, b in zip(points, points[1:]):
        total += math.hypot(b['x'] - a['x'], b['y'] - a['y'])
    return total


def cityflow_topology(path):
    """successors[node] = [(to_node, length, speed, lane_count)], plus signals."""
    with open(path) as handle:
        roadnet = json.load(handle)
    successors = {}
    for road in roadnet['roads']:
        lanes = road.get('lanes', []) or []
        speeds = [lane.get('maxSpeed') for lane in lanes if lane.get('maxSpeed')]
        successors.setdefault(road['startIntersection'], []).append((
            road['endIntersection'],
            _polyline_length(road['points']),
            min(speeds) if speeds else DEFAULT_SPEED,
            len(lanes),
        ))
    graph = build_index_intersection_map_cityflow(path)
    return successors, graph, lambda node_id: node_id


def sumo_topology(path):
    import sumolib

    net = sumolib.net.readNet(path)
    successors = {}
    for edge in net.getEdges():
        successors.setdefault(edge.getFromNode().getID(), []).append((
            edge.getToNode().getID(),
            edge.getLength(),
            edge.getSpeed() or DEFAULT_SPEED,
            edge.getLaneNumber(),
        ))
    graph = build_index_intersection_map_sumo(path)
    # The SUMO graph builder keys signals by the id with any 'GS_' prefix
    # stripped; raw junction ids keep it. Same rule as common/utils.py.
    return successors, graph, lambda node_id: node_id.replace('GS_', '', 1)


def contract_with_geometry(successors, node_id2idx, strip, max_length=None):
    """Shortest-path contraction, carrying the geometry of the path.

    Same edge relation as ``contract_uncontrolled`` -- a directed path from one
    signal to another through unsignalised junctions -- but Dijkstra by length,
    so an edge gets the geometry of the *shortest* path when several exist.
    That is the one that carries the platoon.

    ``max_length`` caps how far the search runs, in metres. Uncapped, a real
    network contracts far past anything that could be called coupling: on
    Ingolstadt21 the mean contracted path is 20 junctions and 1.5 km, which is
    a route across the city, not a link between neighbouring signals. The cap
    is the "hop or distance limit" PROGRESS.md sec 6 (n) left the door open to.
    """
    signals = set(node_id2idx)
    edges = {}
    for raw_src in successors:
        src_key = strip(raw_src)
        if src_key not in signals:
            continue
        # (cumulative length, node, min speed, first-hop lanes, hops)
        heap = [(0.0, raw_src, float('inf'), 0, 0)]
        best = {raw_src: 0.0}
        while heap:
            dist, node, min_speed, first_lanes, hops = heappop(heap)
            if dist > best.get(node, float('inf')):
                continue
            key = strip(node)
            if node != raw_src and key in signals:
                # reached the next signal: record and stop expanding, so paths
                # are contracted rather than chained through signals
                pair = (src_key, key)
                if key != src_key and dist < edges.get(pair, (float('inf'),))[0]:
                    edges[pair] = (dist, min_speed, first_lanes, hops)
                continue
            for to_node, length, speed, lanes in successors.get(node, ()):
                nxt = dist + length
                if max_length is not None and nxt > max_length:
                    continue
                if nxt >= best.get(to_node, float('inf')):
                    continue
                best[to_node] = nxt
                heappush(heap, (
                    nxt, to_node,
                    min(min_speed, speed),
                    lanes if hops == 0 else first_lanes,
                    hops + 1,
                ))
    return edges


def summarise(values):
    if not values:
        return 0.0, 0.0, 0.0, 0.0
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    sd = math.sqrt(var)
    return min(values), mean, max(values), (sd / mean if mean else 0.0)


def analyse(network, world, max_length=None):
    path = roadnet_path(network)
    if not os.path.exists(path):
        return {'network': network, 'error': 'roadnet missing: %s' % path}

    if world == 'cityflow':
        successors, graph, strip = cityflow_topology(path)
    else:
        successors, graph, strip = sumo_topology(path)

    node_id2idx = graph['node_id2idx']
    n_nodes = len(node_id2idx)

    edges = contract_with_geometry(successors, node_id2idx, strip, max_length)

    # Cross-check against the repo's own contraction: identical pair sets, or
    # this script is measuring a graph nothing else uses.
    signal_raw = {node for node in successors if strip(node) in node_id2idx}
    for node in list(successors):
        for to_node, *_ in successors[node]:
            if strip(to_node) in node_id2idx:
                signal_raw.add(to_node)
    plain_succ = {k: [t for t, *_ in v] for k, v in successors.items()}
    reference = contract_uncontrolled(plain_succ, signal_raw, node_id2idx, strip=strip)
    ref_pairs = {(int(s), int(d)) for s, d in reference}
    our_pairs = {(node_id2idx[s], node_id2idx[d]) for s, d in edges}
    # Capped, this script deliberately keeps a subset of the repo's edges.
    agrees = ref_pairs == our_pairs if max_length is None else our_pairs <= ref_pairs

    in_degree = [0] * n_nodes
    for (_src, dst) in edges:
        in_degree[node_id2idx[dst]] += 1

    lengths = [geo[0] for geo in edges.values()]
    taus = [geo[0] / geo[1] for geo in edges.values()]
    lane_counts = [float(geo[2]) for geo in edges.values()]
    hops = [float(geo[3]) for geo in edges.values()]

    return {
        'network': network,
        'world': world,
        'nodes': n_nodes,
        'edges': len(edges),
        'mean_in_degree': sum(in_degree) / n_nodes if n_nodes else 0.0,
        'zero_degree': sum(1 for d in in_degree if d == 0),
        'agrees_with_repo_contraction': agrees,
        'ref_edges': len(ref_pairs),
        'tau': summarise(taus),
        'length': summarise(lengths),
        'lanes': summarise(lane_counts),
        'hops': summarise(hops),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--networks', nargs='*', default=None,
                        help='subset of network names; default is all reported ones')
    parser.add_argument('--min-degree', type=float, default=2.0)
    parser.add_argument('--min-tau-cv', type=float, default=0.2)
    parser.add_argument('--max-length', type=float, default=None,
                        help='cap the contracted path length, in metres')
    args = parser.parse_args()

    wanted = NETWORKS
    if args.networks:
        keep = set(args.networks)
        wanted = [(n, w) for n, w in NETWORKS if n in keep]

    rows = []
    for network, world in wanted:
        try:
            rows.append(analyse(network, world, args.max_length))
        except Exception as exc:  # keep going; one bad roadnet should not hide the rest
            rows.append({'network': network, 'error': '%s: %s' % (type(exc).__name__, exc)})

    header = ('network', 'nodes', 'edges', 'deg', 'deg0', 'tau_cv', 'tau_mean',
              'len_cv', 'len_mean', 'lane_cv', 'hop_mean', 'chk', 'verdict')
    print('%-20s %5s %6s %6s %5s %7s %9s %7s %9s %7s %8s %4s %s' % header)
    print('-' * 118)
    for row in rows:
        if 'error' in row:
            print('%-20s %s' % (row['network'], row['error']))
            continue
        tau_cv = row['tau'][3]
        passes = row['mean_in_degree'] >= args.min_degree and tau_cv >= args.min_tau_cv
        print('%-20s %5d %6d %6.2f %5d %7.3f %9.1f %7.3f %9.1f %7.3f %8.2f %4s %s' % (
            row['network'], row['nodes'], row['edges'], row['mean_in_degree'],
            row['zero_degree'], tau_cv, row['tau'][1], row['length'][3],
            row['length'][1], row['lanes'][3], row['hops'][1],
            'ok' if row['agrees_with_repo_contraction'] else 'DIFF',
            'PASS' if passes else 'fail'))

    print()
    print('pass condition: mean in-degree >= %.1f and CV(tau) >= %.2f'
          % (args.min_degree, args.min_tau_cv))
    print('tau = shortest contracted path length / min speed limit along it, in seconds.')
    print('chk = this script\'s edge set equals common.utils.contract_uncontrolled\'s.')


if __name__ == '__main__':
    main()
