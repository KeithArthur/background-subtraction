import scipy.sparse as sp
import numpy as np
import utils

import platform
project_float = np.float64 if '64' in platform.architecture()[0] else np.float32

def build_graph(frame_dimensions, batch_dimensions):
    frame_height_m, frame_width_n = frame_dimensions
    batch_height = min(batch_dimensions[0], frame_height_m)
    batch_width = min(batch_dimensions[1], frame_width_n)
    num_x = frame_height_m - batch_height + 1
    num_y = frame_width_n - batch_width + 1
    num_groups = num_x * num_y
    graph = {'eta_g': np.ones(num_groups, dtype=project_float),
             'groups': sp.csc_matrix((num_groups, num_groups),
                                     dtype=np.bool),
             'groups_var': sp.lil_matrix((frame_height_m * frame_width_n, num_groups), dtype=np.bool)}
    for i in range(0, num_groups):
        indX = i / num_y
        indY = i % num_y
        
        for x in range(indX, indX + batch_height):
            for y in range(indY, indY + batch_width):
                graph['groups_var'][utils.index2d_to_1d(x,y,frame_dimensions), i] = True

    graph['groups_var'] = sp.csc_matrix(graph['groups_var'])
    return graph
