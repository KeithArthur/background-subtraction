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
        'positions': [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        'deltas': [[], [], [], []]
    })

def test__init_missing_trajectories_2():
    """Mutates trajectories to create new trajectories but does not update existing ones"""
    flow = np.array([[[1.0, 0.0], [-1.0, 0.0]],
                     [[1.0, 0.0], [-1.0, 0.0]]])
    trajectories = {'positions': [np.array(coll) for coll in [[1.0, 0.0]]],
                    'deltas': [[np.array(coll)] for coll in [[1.0, 0.0]]]}
    m._init_missing_trajectories(flow, trajectories)
    assert_equal(trajectories, {
        'positions': [[1.0, 0.0], [0.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        'deltas': [[[1.0, 0.0]], [np.nan], [np.nan], [np.nan]]
    })

def test__init_missing_trajectories_3():
    """Mutates trajectories to create new trajectories but does not update
existing ones. Adds nans for frames where the trajectory did not
exist."""
    flow = np.array([[[1.0, 0.0], [-1.0, 0.0]],
                     [[1.0, 0.0], [-1.0, 0.0]]])
    trajectories = {'positions': [np.array(coll) for coll in [[1.0, 0.0]]],
                    'deltas': [[[1.0, 0.0], [0.0, 0.0]]]}
    m._init_missing_trajectories(flow, trajectories)
    assert_equal(trajectories, {
        'positions': [[1.0, 0.0], [0.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        'deltas': [[[1.0, 0.0], [0.0, 0.0]], [np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]]
    })

def test__update_trajectories():
    """Mutates trajectories to update existing trajectories but does not initialize new ones"""
    flow = np.array([[[1.0, 0.0], [-1.0, 0.0]],
                     [[1.0, 0.0], [-1.0, 0.0]]])
    trajectories = {'positions': [[1.0, 0.0]],
                    'deltas': [[[1.0, 0.0]]]}
    m._update_trajectories(flow, trajectories, [2, 2])
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

def test_calc_trajectories_1():
    """Returns trajectories for each pair of flows"""
    forward_flows = [np.array([[[1.0, 0.0], [-1.0, 0.0]],
                               [[1.0, 0.0], [-1.0, 0.0]]]),
                     np.array([[[1.0, 0.0], [-1.0, 0.0]],
                               [[1.0, 0.0], [-1.0, 0.0]]]),
                     np.array([[[1.0, 0.0], [-1.0, 0.0]],
                               [[1.0, 0.0], [-1.0, 0.0]]])]
    backward_flows = [np.array([[[1.0, 0.0], [-1.0, 0.0]],
                                [[1.0, 0.0], [-1.0, 0.0]]]),
                      np.array([[[1.0, 0.0], [-1.0, 0.0]],
                                [[1.0, 0.0], [-1.0, 0.0]]]),
                      np.array([[[1.0, 0.0], [-1.0, 0.0]],
                                [[1.0, 0.0], [-1.0, 0.0]]])]
    trajectories = m.calc_trajectories(forward_flows, backward_flows, [2, 2])
    assert_equal(trajectories['deltas'], [[[1.0, 0.0], [-1.0, 0.0], [1.0, 0.0]],
                                          [[-1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]],
                                          [[1.0, 0.0], [-1.0, 0.0], [1.0, 0.0]],
                                          [[-1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]]])
    assert_equal(trajectories['positions'], [[1.0, 0.0], [0.0, 0.0], [1.0, 1.0], [0.0, 1.0]])


def test_calc_trajectories_2():
    """Returns trajectories for each pair of flows, terminating any occluded trajectories"""
    forward_flows = [np.array([[[1.0, 0.0], [-1.0, 0.0]],
                               [[1.0, 0.0], [-1.0, 0.0]]]),
                     np.array([[[1.0, 0.0], [-1.0, 0.0]],
                               [[1.0, 0.0], [-1.0, 0.0]]]),
                     np.array([[[1.0, 0.0], [-1.0, 0.0]],
                               [[1.0, 0.0], [-1.0, 0.0]]])]
    backward_flows = [np.array([[[1.0, 0.0], [-1.0, 0.0]],
                                [[1.0, 0.0], [-1.0, 0.0]]]),
                      np.array([[[-1.0, 0.0], [-1.0, 0.0]],
                                [[1.0, 0.0], [-1.0, 0.0]]]),
                      np.array([[[1.0, 0.0], [-1.0, 0.0]],
                                [[1.0, 0.0], [-1.0, 0.0]]])]
    trajectories = m.calc_trajectories(forward_flows, backward_flows, [2, 2])
    assert_equal(trajectories['deltas'], [[[1.0, 0.0], [-1.0, 0.0]],
                                          [[-1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]],
                                          [[1.0, 0.0], [-1.0, 0.0], [1.0, 0.0]],
                                          [[-1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]],
                                          [np.nan, np.nan, [1.0, 0.0]]])
    assert_equal(trajectories['positions'], [[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])


def test_deltas_to_positions():
    trajectories = {'deltas': [np.array(coll) for coll in [[[-1.0, 0.0], [-1.0, 1.0]], [[0.0, -1.0], [0.0, 1.0]]]],
                    'positions': [np.array(coll) for coll in [[0.0, 0.0], [0.0, 1.0]]]}
    assert_equal(m.deltas_to_positions(trajectories), [[[2.0, -1.0], [1.0, 0.0], [0.0, 0.0]],
                                                        [[0.0, 1.0], [0.0, 2.0], [0.0, 1.0]]])

def test_calc_motion_saliencies():
    """returns a list of the motion saliencies"""
    trajectories = {'positions': [np.array(col) for col in [[1.0, 0.0], [0.0, 0.0], [0.0, 1.0], [1.0, 1.0]]],
                    'deltas': [[[1.0, 0.0], [0.0, 0.0]], [np.nan, [0.0, 0.0]], [np.nan, [0.0, 0.0]], [np.nan, [0.0, 0.0]]]}
    assert_equal(m.calc_motion_saliencies(trajectories), [1.0, 0, 0, 0])


def test_get_pixel_trajectory_lookup():
    trajectories = {'deltas': [[[-1.0, 0.0]], [[-1.0, 0.0]], [[1.0, 0.0]], [[1.0, 0.0]]],
                    'positions': [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]}
    pixel_trajectory_lookup = m.get_pixel_trajectory_lookup(trajectories, (2, 2, 2))
    assert_equal(pixel_trajectory_lookup, [[[2, 0],
                                            [3, 1]],
                                           [[0, 2],
                                            [1, 3]]])

