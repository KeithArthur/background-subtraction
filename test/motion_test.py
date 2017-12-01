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


def test__deltas_to_positions():
    trajectories = {'deltas': [np.array(coll) for coll in [[[-1.0, 0.0], [-1.0, 1.0]], [[0.0, -1.0], [0.0, 1.0]]]],
                    'positions': [np.array(coll) for coll in [[0.0, 0.0], [0.0, 1.0]]]}
    assert_equal(m._deltas_to_positions(trajectories), [[[2.0, -1.0], [1.0, -1.0], [0.0, 0.0]],
                                                        [[0.0, 1.0], [0.0, 0.0], [0.0, 1.0]]])

def test__is_salient():
    assert m._is_salient(np.array([[-1.0, 0.0], [-1.0, 1.0]]))
    assert not m._is_salient(np.array([[0.0, -1.0], [0.0, 0.0]]))

def test__get_inconsistent_trajectory_nums():
    trajectories = {'deltas': [np.array(coll) for coll in [[[0.0, -1.0], [0.0, 0.0]], [[-1.0, 0.0], [-1.0, 1.0]]]]}
    inconsistent_trajectory_nums = m._get_inconsistent_trajectory_nums(trajectories)
    assert inconsistent_trajectory_nums == [0]

def test__calc_trajectory_saliencies_1():
    """returns a list of the motion saliencies"""
    trajectories = {'positions': [np.array(col) for col in [[10.0, 0.0], [0.0, 0.0], [0.0, 1.0], [1.0, 1.0]]],
                    'deltas': [[[5.0, 0.0], [5.0, 0.0]], [np.nan, [0.0, 0.0]], [np.nan, [0.0, 0.0]], [np.nan, [0.0, 0.0]]]}
    assert_equal(m._calc_trajectory_saliencies(trajectories), [10.0, 0, 0, 0])


def test__calc_trajectory_saliencies_2():
    """appends 0 for inconsistent motion"""
    trajectories = {'positions': [np.array(col) for col in [[1.0, 0.0], [0.0, 0.0], [0.0, 1.0], [1.0, 1.0]]],
                    'deltas': [[[1.0, 0.0], [0.0, 0.0]], [np.nan, [0.0, 0.0]], [np.nan, [0.0, 0.0]], [np.nan, [0.0, 0.0]]]}
    assert_equal(m._calc_trajectory_saliencies(trajectories), [0.0, 0, 0, 0])


def test__get_pixel_trajectory_lookup_1():
    trajectories = {'deltas': [[[-1.0, 0.0]], [[-1.0, 0.0]], [[1.0, 0.0]], [[1.0, 0.0]]],
                    'positions': [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]}
    pixel_trajectory_lookup = m._get_pixel_trajectory_lookup(trajectories, (2, 2, 2))
    assert_equal(pixel_trajectory_lookup, [[[2, 0],
                                            [3, 1]],
                                           [[0, 2],
                                            [1, 3]]])


def test__get_pixel_trajectory_lookup_2():
    """-1 wherever a trajectory does not pass through. This is to set a saliency of 0 later."""
    trajectories = {'deltas': [[[-1.0, 0.0]], [[-1.0, 0.0]], [[1.0, 0.0]]],
                    'positions': [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]]}
    pixel_trajectory_lookup = m._get_pixel_trajectory_lookup(trajectories, (2, 2, 2))
    assert_equal(pixel_trajectory_lookup, [[[2, 0],
                                            [-1, 1]],
                                           [[0, 2],
                                            [1, -1]]])

def test__get_pixel_trajectory_lookup_3():
    trajectories = {'deltas': [[[1.0, 0.0]]],
                    'positions': [[1.0, 0.0]]}
    pixel_trajectory_lookup = m._get_pixel_trajectory_lookup(trajectories, (2, 2, 2))
    assert_equal(pixel_trajectory_lookup, [[[0, -1],
                                            [-1, -1]],
                                           [[-1, 0],
                                            [-1, -1]]])

def test__get_pixel_saliencies_1():
    trajectory_saliencies = [1.0, 2.0, 0.5, 3.0]
    pixel_trajectory_lookup = [[[2, 0], [3, 1]]]
    pixel_saliencies = m._get_pixel_saliencies(trajectory_saliencies, pixel_trajectory_lookup)
    assert_equal(pixel_saliencies, [[[0.5, 1.0], [3.0, 2.0]]])

def test__get_pixel_saliencies_2():
    trajectory_saliencies = [1.0, 2.0, 0.5, 3.0]
    pixel_trajectory_lookup = [[[2, -1], [3, 1]]]
    pixel_saliencies = m._get_pixel_saliencies(trajectory_saliencies, pixel_trajectory_lookup)
    assert_equal(pixel_saliencies, [[[0.5, 0.0], [3.0, 2.0]]])


def test_set_groups_saliencies_1():
    groups = [{'frame': 0, 'elems': [[0, 0]]}]
    trajectories = {'deltas': [np.array(coll) for coll in [[[0.0, 1.0]]]],
                    'positions': [np.array(coll) for coll in [[0.0, 1.0]]]}
    video_data_dimensions = (2, 2, 2)
    m.set_groups_saliencies(groups, trajectories, video_data_dimensions)
    assert_equal(groups, [{'frame': 0, 'elems': [[0, 0]], 'salience': 1.0}])


def test_set_groups_saliencies_2():
    """Note that the elem coords are [row, col] and trajectory positions
are [x, y] which is the reverse so the second group has inconsistent
motion and thus a salience of -1."""
    groups = [{'frame': 0, 'elems': [[0, 0]]}, {'frame': 1, 'elems': [[0, 1]]}]
    trajectories = {'deltas': [np.array(coll) for coll in [[[0.0, 1.0]]]],
                    'positions': [np.array(coll) for coll in [[0.0, 1.0]]]}
    video_data_dimensions = (2, 2, 2)
    m.set_groups_saliencies(groups, trajectories, video_data_dimensions)
    assert_equal(groups, [{'frame': 0, 'elems': [[0, 0]], 'salience': 1.0},
                          {'frame': 1, 'elems': [[0, 1]], 'salience': 0.0}])


def test_set_groups_saliencies_3():
    groups = [{'frame': 0, 'elems': [[0, 0]]}, {'frame': 1, 'elems': [[1, 0]]}]
    trajectories = {'deltas': [np.array(coll) for coll in [[[0.0, 1.0]]]],
                    'positions': [np.array(coll) for coll in [[0.0, 1.0]]]}
    video_data_dimensions = (2, 2, 2)
    m.set_groups_saliencies(groups, trajectories, video_data_dimensions)
    assert_equal(groups, [{'frame': 0, 'elems': [[0, 0]], 'salience': 1.0},
                          {'frame': 1, 'elems': [[1, 0]], 'salience': 1.0}])


def test_set_groups_saliencies_4():
    groups = [{'frame': 0, 'elems': [[0, 0], [0, 1]]}, {'frame': 1, 'elems': [[1, 0], [1, 1]]}]
    trajectories = {'deltas': [np.array(coll) for coll in [[[0.0, 1.0]], [[1.0, 0.0]]]],
                    'positions': [np.array(coll) for coll in [[0.0, 1.0], [1.0, 1.0]]]}
    video_data_dimensions = (2, 2, 2)
    m.set_groups_saliencies(groups, trajectories, video_data_dimensions)
    assert_equal(groups, [{'frame': 0, 'elems': [[0, 0], [0, 1]], 'salience': 0.5},
                          {'frame': 1, 'elems': [[1, 0], [1, 1]], 'salience': 1.0}])


def test_set_regularization_lambdas():
    groups = [{'frame': 0, 'elems': [[0, 0], [0, 1]], 'salience': 0.5},
              {'frame': 1, 'elems': [[1, 0], [1, 1]], 'salience': 1.0}]
    video_data_dimensions = (2, 2, 2)
    m.set_regularization_lambdas(groups, video_data_dimensions)
    assert_equal(groups, [{'frame': 0, 'elems': [[0, 0], [0, 1]], 'salience': 0.5, 'regularization_lambda': 1.0/np.sqrt(2) * 0.1},
                          {'frame': 1, 'elems': [[1, 0], [1, 1]], 'salience': 1.0, 'regularization_lambda': 0.5/np.sqrt(2) * 0.1}])
