import numpy as np
import numpy.linalg as la

from linalg import dual_norm, shrink

def _calc_background_L(frames_D, dual_Y, dual_mu, background_L, foreground_S):
    background_G_L = frames_D - foreground_S + dual_Y / dual_mu
    return shrink(background_G_L, 1.0 / dual_mu)

def _calc_foreground_S(frames_D, dual_Y, dual_mu, background_L, foreground_S):
    foreground_G_S = frames_D - background_L + dual_Y / dual_mu
    return min_cost_flow(foreground_G_S)

def _calc_Y(frames_D, dual_Y, dual_mu, background_L, foreground_S):
    return dual_Y + dual_mu * (frames_D - background_L - foreground_S)

def inexact_alm_lsd(frames_D, graph, max_iterations=100):
    alm_penalty_scalar_rho = 1.5
    tolerance = 1e-7
    num_pixels_n, num_frames_p = frames_D.shape
    regularization_lambda = 1.0 / np.sqrt(num_pixels_n)
    dual_mu = 12.5 / la.norm(frames_D, ord=2)
    dual_Y = frames_D / dual_norm(frames_D, regularization_lambda)
    foreground_S = np.zeros_like(frames_D) # E in reference code
    background_L = np.zeros_like(frames_D) # A in reference code
    while not_converged:
        foreground_S = _calc_background_L(frames_D, dual_Y, dual_mu, background_L, foreground_S)
        background_L = _calc_foreground_S(frames_D, dual_Y, dual_mu, background_L, foreground_S)
        dual_Y = _calc_Y(frames_D, dual_Y, dual_mu, background_L, foreground_S)
        dual_mu = alm_penalty_scalar_rho * dual_mu