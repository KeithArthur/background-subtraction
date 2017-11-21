import numpy as np
from numpy.testing import assert_allclose as eq
from numpy.testing import assert_equal

import motion as m


def test__get_remaining_indices():
    """Returns a set difference of the indices for the specified matrix shape"""
    assert m._get_remaining_indices([np.array(coll) for coll in [[1.0, 1.0], [0.0, 0.0]]], (2, 2)) == [[0, 1], [1, 0]]

def test__init_missing_trajectories_1():
    """Mutates the trajectories to initialize trajectories"""
    flow = np.array([[[0.0, 1.0], [-1.0, 0.0]],
                     [[1.0, 0.0], [0.0, -1.0]]])
    trajectories = {'positions': [],
                    'deltas': []}
    m._init_missing_trajectories(flow, trajectories)
    assert_equal(trajectories, {
        'positions': [[0.0, 1.0], [0.0, 0.0], [1.0, 1.0], [1.0, 0.0]],
        'deltas': [[[0.0, 1.0]], [[-1.0, 0.0]], [[1.0, 0.0]], [[0.0, -1.0]]]
    })

def test__init_missing_trajectories_2():
    """Mutates trajectories to create new trajectories but does not update existing ones"""
    flow = np.array([[[1.0, 0.0], [-1.0, 0.0]],
                     [[1.0, 0.0], [-1.0, 0.0]]])
    trajectories = {'positions': [np.array(coll) for coll in [[1.0, 0.0]]],
                    'deltas': [[np.array(coll)] for coll in [[1.0, 0.0]]]}
    m._init_missing_trajectories(flow, trajectories)
    assert_equal(trajectories, {
        'positions': [[1.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        'deltas': [[[1.0, 0.0]], [np.nan, [1.0, 0.0]], [np.nan, [1.0, 0.0]], [np.nan, [-1.0, 0.0]]]
    })

def test__update_trajectories():
    """Mutates trajectories to update existing trajectories but does not initialize new ones"""
    flow = np.array([[[1.0, 0.0], [-1.0, 0.0]],
                     [[1.0, 0.0], [-1.0, 0.0]]])
    trajectories = {'positions': [[1.0, 0.0]],
                    'deltas': [[[1.0, 0.0]]]}
    m._update_trajectories(flow, trajectories)
    assert_equal(trajectories, {
        'positions': [[0.0, 0.0]],
        'deltas': [[[1.0, 0.0], [-1.0, 0.0]]]
    })

def test__end_occluded_trajectories():
    """Mutates trajectories to remove occluded trajectories. Also returns the removed trajectories"""
    forward_flow = np.array([[[1.0, 0.0], [-1.0, 0.0]],
                             [[1.0, 0.0], [-1.0, 0.0]]])
    backward_flow = np.array([[[-1.0, 0.0], [-1.0, 0.0]],
                              [[1.0, 0.0], [-1.0, 0.0]]])
    trajectories = {'positions': [[0.0, 0.0]],
                    'deltas': [[[-1.0, 0.0]]]}
    complete_trajectories = m._end_occluded_trajectories(forward_flow, backward_flow, trajectories)
    assert_equal(trajectories, {
        'positions': [],
        'deltas': []
    })
    assert_equal(complete_trajectories, {
        'positions': [[0.0, 0.0]],
        'deltas': [[[-1.0, 0.0]]]
    })
