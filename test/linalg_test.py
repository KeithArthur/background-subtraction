from pytest import approx
import numpy as np

import linalg

def test_dual_norm():
    assert approx(linalg.dual_norm(np.zeros((5, 5)), 1.0)) == 0.0
    assert approx(linalg.dual_norm(np.ones((5, 5)), 1.0)) == 5.0
    assert approx(linalg.dual_norm(np.eye(5), 1.0)) == 1.0
    assert approx(linalg.dual_norm(np.eye(5) * 10.0, 1.0)) == 10.0
    assert approx(linalg.dual_norm(np.ones((5, 5)), 2.0)) == 5.0
    assert approx(linalg.dual_norm(np.eye(5) * 10.0, 11.0)) == 10.0
    assert approx(linalg.dual_norm(np.ones((5, 5)) * 10.0, 11.0)) == 50.0
