import numpy as np
from numpy.testing import assert_allclose as eq

import utils as u

def test_extend_dict():
    animals = {'cats': ['sam', 'john'],
               'dogs': ['max']}
    u.extend_dict(animals, {'cats': ['tim'], 'dogs': ['lou']})
    assert animals == {'cats': ['sam', 'john', 'tim'],
                       'dogs': ['max', 'lou']}
