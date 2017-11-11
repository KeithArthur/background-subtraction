import scipy.sparse as sp
import numpy as np

def build_graph(frame_dimensions, batch_dimensions):
    frame_height_m, frame_width_n = frame_dimensions
    batch_height = min(batch_dimensions[0], frame_height_m)
    batch_width = min(batch_dimensions[1], frame_width_n)
    num_x = frame_height_m - batch_height + 1
    num_y = frame_width_n - batch_width + 1
    num_in_group = num_x * num_y
    graph = {'eta_g': np.ones(num_in_group),
             'groups': sp.csc_matrix(np.zeros((num_in_group, num_in_group)),
                                     dtype=np.bool),
             'groups_var': sp.csc_matrix(np.zeros((frame_height_m * frame_width_n, num_in_group)), dtype=np.bool)}
    for i in range(0, num_in_group):
        indiMatrix = np.zeros((frame_height_m, frame_width_n))
        indX = i % num_x
        indY = i / num_x
        indiMatrix[indX:indX+batch_height, indY:indY+batch_width] = True
        graph['groups_var'][np.where(indiMatrix.ravel()), i] = True
    return graph
