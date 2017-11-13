from pytest import approx
from numpy.testing import assert_allclose as eq
import numpy as np
import scipy.sparse as sp

import lsd_operations as lsd
from graph import build_graph

def test_dual_norm():
    assert approx(lsd.dual_norm(np.zeros((5, 5)), 1.0)) == 0.0
    assert approx(lsd.dual_norm(np.ones((5, 5)), 1.0)) == 5.0
    assert approx(lsd.dual_norm(np.eye(5), 1.0)) == 1.0
    assert approx(lsd.dual_norm(np.eye(5) * 10.0, 1.0)) == 10.0
    assert approx(lsd.dual_norm(np.ones((5, 5)), 2.0)) == 5.0
    assert approx(lsd.dual_norm(np.eye(5) * 10.0, 11.0)) == 10.0
    assert approx(lsd.dual_norm(np.ones((5, 5)) * 10.0, 11.0)) == 50.0

def test__soft_thresh():
    eq(lsd._soft_thresh(np.ones(5), 0.9),
       0.1 * np.ones(5))
    eq(lsd._soft_thresh(np.ones(5), 1),
       np.zeros(5))
    eq(lsd._soft_thresh(np.array([0.1, 2, 0.2]), 1),
       np.array([0, 1.0, 0]))

def test_shrink():
    eq(lsd.shrink(np.eye(5), 1),
       np.zeros((5, 5)))
    eq(lsd.shrink(np.ones((5, 5)), 1),
       np.ones((5, 5)) * 0.8)
    eq(lsd.shrink(np.array([[1, 2], [3, 4]]), 1),
       np.array([[ 1.04053125,  1.47651896],
                 [ 2.3521747 ,  3.33774745]]))

def test_min_cost_flow_l1():
    input_signal_U = np.asfortranarray(np.array([[1.0, 2.0], [3.0, 4.0]]), dtype=np.float32)
    groups_dimensions = (input_signal_U.shape[0], input_signal_U.shape[0])
    singleton_graph = {'eta_g': np.ones(input_signal_U.shape[0], dtype=np.float32),
                       'groups': sp.csc_matrix(np.zeros(groups_dimensions), dtype=np.bool),
                       'groups_var': sp.csc_matrix(np.eye(input_signal_U.shape[0], dtype=np.bool), dtype=np.bool)}
    prox_l1 = lsd.min_cost_flow(input_signal_U, singleton_graph, 2.1)
    eq(prox_l1, np.array([[0.0, 0.0], [0.9, 1.9]], dtype=np.float32), rtol=1e-6)

def test_min_cost_flow():
    input_signal_U = np.asfortranarray(np.array([[1.0, 2.0], [3.0, 4.0]]), dtype=np.float32)
    groups_dimensions = (1, 1)
    graph = {'eta_g': np.ones(1, dtype=np.float32),
             'groups': sp.csc_matrix(np.zeros(groups_dimensions), dtype=np.bool),
             'groups_var': sp.csc_matrix(np.ones((input_signal_U.shape[0], 1), dtype=np.bool), dtype=np.bool)}
    prox = lsd.min_cost_flow(input_signal_U, graph, 0.1)
    eq(prox, np.array([[1.0, 2.0], [2.9, 3.9]], dtype=np.float32), rtol=1e-6)

def test_min_cost_flow_pixel():
    input_signal_U = np.asfortranarray(np.array([[1.0, 2.0, 3.0, 4.0, 5.0],
                                                 [2.0, 3.0, 2.0, 1.0, 4.0],
                                                 [5.0, 2.0, 2.0, 4.0, 3.0],
                                                 [8.0, 8.0, 3.0, 2.0, 5.0],
                                                 [2.0, 3.0, 2.0, 1.0, 4.0],
                                                 [5.0, 2.0, 2.0, 4.0, 3.0]]), dtype=np.float32)
    graph = build_graph([3, 2], [2, 2])
    prox = lsd.min_cost_flow(input_signal_U, graph, 0.1)
    eq(prox, np.array([[1.0, 2.0, 2.9, 3.95, 4.9],
                       [2.0, 3.0, 2.0, 1.0, 4.0],
                       [5.0, 2.0, 2.0, 3.95, 3.0],
                       [7.8, 7.8, 2.9, 1.9, 4.9],
                       [2.0, 3.0, 2.0, 1.0, 4.0],
                       [5.0, 2.0, 2.0, 4.0, 3.0]], dtype=np.float32))
