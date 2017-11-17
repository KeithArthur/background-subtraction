import numpy as np
import numpy.linalg as la
import spams

import platform
project_float = np.float64 if '64' in platform.architecture()[0] else np.float32

def dual_norm(M, regularization_lambda):
    largest_eigenvalue = la.norm(M, ord=2)
    largest_val = np.max(M)
    return max(largest_eigenvalue, largest_val / regularization_lambda)

def _soft_thresh(S, threshold):
    return [max(0, x - threshold) for x in S]

def shrink(M, threshold):
    [U, S, V] = la.svd(M, full_matrices=False)
    return np.dot(U, np.dot(np.diag(_soft_thresh(S, threshold)),
                            V))

def min_cost_flow(input_signal_U, graph, lambda1):
    return spams.proximalGraph(np.asfortranarray(input_signal_U, dtype=project_float),
                               graph,
                               False,
                               numThreads=-1,
                               lambda1=lambda1,
                               regul='graph',
                               verbose=False,
                               pos=False,
                               intercept=False);
