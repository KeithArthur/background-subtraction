from pytest import approx
from numpy.testing import assert_allclose as eq
import numpy as np

import lsd_operations as lsd

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

# def test_min_cost_flow()
