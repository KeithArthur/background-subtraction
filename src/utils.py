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

def index2d_to_1d(x,y, f_dim):
    return x + y * f_dim[0]

def index1d_to_2d(x, f_dim):
    return x % f_dim[0], x / f_dim[0]

def find_groups(fg_images, frame_num, f_dim):
    groups = []
    search_dir = [[+1, 0], [0, +1], [-1, 0], [0, -1]]
    
    for fn in range(frame_num):
        visit = np.zeros(f_dim)
        for x0 in range(f_dim[0]):
            for y0 in range(f_dim[1]):
                if( fg_images[fn][x0, y0] <= 0 or visit[x0, y0] == 1 ): continue
                gr_list, queue = [], [[x0,y0]]
                visit[x0, y0] = 1
                
                while(len(queue) > 0):
                    x, y = queue.pop()
                    gr_list.append([x,y])
                    for dx, dy in search_dir:
                        if( x+dx >= 0 and y+dy>=0 and x+dx < f_dim[0] and 
                           y+dy < f_dim[1] and visit[x+dx, y+dy] == 0 and
                           fg_images[fn][x+dx, y+dy] > 0 ):
                            queue.append([x+dx, y+dy])
                            visit[x+dx, y+dy] = 1
    
                groups.append({'frame':fn, 'elems':gr_list})
                
    return groups