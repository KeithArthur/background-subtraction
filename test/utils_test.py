import numpy as np
from numpy.testing import assert_allclose as eq

import utils as u

def test_extend_dict():
    animals = {'cats': ['sam', 'john'],
               'dogs': ['max']}
    u.extend_dict(animals, {'cats': ['tim'], 'dogs': ['lou']})
    assert animals == {'cats': ['sam', 'john', 'tim'],
                       'dogs': ['max', 'lou']}

def test_enumerate_pairs_with_order():
    assert u.enumerate_pairs_with_order([1, 2, 3, 4]) == [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
