import numpy as np
from itertools import tee, islice, chain, izip

def prev_cur_next(some_iterable):
    prevs, items, nexts = tee(some_iterable, 3)
    prevs = chain([None], prevs)
    nexts = chain(islice(nexts, 1, None), [None])
    return izip(prevs, items, nexts)

def left_pad(coll, elem, num):
    if isinstance(coll, np.ndarray):
        return np.append([elem for i in range(num)], coll)
    else:
        return [elem for i in range(num)] + coll

def extend_dict(coll_to, coll_from):
    for key, val_list in coll_from.items():
        coll_to[key].extend(val_list)

def enumerate_pairs_with_order(coll):
    to_enumerate = []
    for index, elem in enumerate(coll[:-1]):
        to_enumerate.extend([(elem, other_elem) for other_elem in coll[index + 1:]])
    return to_enumerate

def index2d_to_1d(x, y, f_dim):
    return x + y * f_dim[0]

def index1d_to_2d(x, f_dim):
    return [x % f_dim[0], x / f_dim[0]]
