"""Regression tests for CoLight's node ordering.

The bug: ``graph['sparse_adj']`` is indexed by graph node index (the order
intersections appear in the converted roadnet), while observations, rewards,
phase_lengths and the action vector the trainer applies are all indexed by
``world.intersections``.  The constructor meant to reconcile them and called
``sorted(generators, key=...)`` without assigning the result -- a no-op, ten
times over.  On the CityFlow grids the two orders coincide and nothing shows;
on Ingolstadt21 all 21 rows are out of place, so every node aggregated the
neighbours of an unrelated intersection.
"""

import unittest

import numpy as np


class _Inter:
    def __init__(self, id_):
        self.id = id_


class _Agent:
    """Just enough of CoLightAgent to exercise the remap."""

    from agent.colight import CoLightAgent
    _world_ordered_adjacency = CoLightAgent._world_ordered_adjacency

    def __init__(self, world_ids, graph_order, edges):
        self.world = type('W', (), {'intersections': [_Inter(i) for i in world_ids]})()
        self.graph = {
            'node_idx2id': {i: n for i, n in enumerate(graph_order)},
            'sparse_adj': np.array(edges, dtype=np.int64),
        }


class AdjacencyOrderTests(unittest.TestCase):
    def test_identity_when_orders_agree(self):
        ids = ['a', 'b', 'c']
        agent = _Agent(ids, ids, [[0, 1], [1, 2]])
        np.testing.assert_array_equal(
            agent._world_ordered_adjacency(), np.array([[0, 1], [1, 2]]))

    def test_edges_follow_identity_not_position(self):
        # graph order (a, b, c) against world order (c, a, b): the edge a->b is
        # graph (0,1) and must come out as world (1,2).
        agent = _Agent(['c', 'a', 'b'], ['a', 'b', 'c'], [[0, 1], [1, 2]])
        out = agent._world_ordered_adjacency()
        np.testing.assert_array_equal(out, np.array([[1, 2], [2, 0]]))

    def test_endpoints_keep_their_identity(self):
        world = ['n3', 'n1', 'n4', 'n2']
        graph = ['n1', 'n2', 'n3', 'n4']
        edges = [[0, 1], [1, 2], [2, 3], [3, 0]]
        agent = _Agent(world, graph, edges)
        out = agent._world_ordered_adjacency()
        for (gs, gt), (ws, wt) in zip(edges, out):
            self.assertEqual(graph[gs], world[ws])
            self.assertEqual(graph[gt], world[wt])

    def test_gs_prefix_is_stripped(self):
        # the SUMO world prefixes some ids with GS_; the graph does not.
        agent = _Agent(['GS_b', 'GS_a'], ['a', 'b'], [[0, 1]])
        np.testing.assert_array_equal(agent._world_ordered_adjacency(), np.array([[1, 0]]))

    def test_missing_node_is_refused_not_silently_dropped(self):
        agent = _Agent(['a', 'b'], ['a', 'ghost'], [[0, 1]])
        with self.assertRaises(ValueError):
            agent._world_ordered_adjacency()


if __name__ == '__main__':
    unittest.main()
