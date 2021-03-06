import numpy as np
from numpy.testing import assert_allclose as eq
from numpy.testing import assert_equal

import sys
sys.path.append("../src/")
import group as g

def test_find_groups():
    test_array = np.array([[[0,0,0,1,1,0,0],
                            [0,0,1,1,0,0,0],
                            [0,0,0,0,1,0,0],
                            [0,0,1,0,1,0,0],
                            [0,1,1,0,1,0,0]],
                        
                          [[0,0,0,1,1,0,0],
                           [0,0,1,0,0,0,0],
                           [0,0,1,0,1,0,0],
                           [0,0,1,0,1,0,0],
                           [0,0,1,0,1,0,0]]])
    
    ginfo = g.find_groups(test_array, 2, [5,7])
    assert_equal(len(ginfo), 6)
    assert_equal(len(ginfo[0]['elems']), 4)
    assert_equal(len(ginfo[1]['elems']), 3)
    assert_equal(len(ginfo[2]['elems']), 3)
    
    assert_equal(ginfo[3]['frame'], 1)
    assert_equal(ginfo[3]['elems'], [[0,3], [0,4]])


def test_keep_only_in_group():
    mat = np.array([[1, 2], [3, 4]])
    group = {'frame': 0, 'elems': [[0, 1], [1, 1]]}
    filtered_mat = g.keep_only_in_group(mat, group['elems'])
    assert_equal(filtered_mat, [[0, 2], [0, 4]])
