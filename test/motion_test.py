import numpy as np
from numpy.testing import assert_allclose as eq
from numpy.testing import assert_equal

import motion as m


def test__get_remaining_indices():
    assert m._get_remaining_indices([np.array(coll) for coll in [[1.0, 1.0], [0.0, 0.0]]], (2, 2)) == [[0, 1], [1, 0]]

def test__init_missing_trajectories():
    flow = np.array([[[1.0, 0.0], [0.0, -1.0]],
                     [[0.0, 1.0], [-1.0, 0.0]]])
    trajectories = {'positions': [np.array(coll) for coll in []],
                    'deltas': [np.array(coll) for coll in []]}
    m._init_missing_trajectories(flow, trajectories)
    assert_equal( trajectories, {
        'positions': [[np.array(coll)] for coll in [[1.0, 0.0], [0.0, 0.0], [1.0, 1.0], [0.0, 1.0]]],
        'deltas': [[[1.0, 0.0]], [[0.0, -1.0]], [[0.0, 1.0]], [[-1.0, 0.0]]]
    })
