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
