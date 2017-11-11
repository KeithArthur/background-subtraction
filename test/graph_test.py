import graph as g

def test_build_graph():
    graph = g.build_graph([5, 5], [3, 3])
    assert graph['groups_var'].shape == (25, 9)
    assert len(graph['groups_var'].nonzero()[0]) == 81
    assert graph.keys() == ['groups_var', 'groups', 'eta_g']
