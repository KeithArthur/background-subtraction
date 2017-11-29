import numpy as np
import numpy.linalg as la

from lsd_operations import dual_norm, shrink, min_cost_flow

import platform
project_float = np.float64 if '64' in platform.architecture()[0] else np.float32

def _calc_background_L(frames_D, dual_Y, dual_mu, foreground_S, rk):
    background_G_L = frames_D - foreground_S + dual_Y / dual_mu
    return shrink(background_G_L, 1.0 / dual_mu, rk)

def _calc_foreground_S(frames_D, dual_Y, dual_mu, background_L, graph, regularization_lambda):
    foreground_G_S = frames_D - background_L + dual_Y / dual_mu
    
    #L1- norm
    # return np.maximum(np.abs(foreground_G_S) - regularization_lambda, 0) * np.sign(foreground_G_S)
    
    #Structured Sparsity 
    return min_cost_flow(foreground_G_S, graph, regularization_lambda / dual_mu)

def _calc_Y(frames_D, dual_Y, dual_mu, background_L, foreground_S):
    return dual_Y + dual_mu * (frames_D - background_L - foreground_S)

def _calc_error(frames_D, background_L, foreground_S):
    distance = frames_D - background_L - foreground_S
    return la.norm(distance,'fro') / la.norm(frames_D,'fro');

def inexact_alm_lsd(frames_D, graph, max_iterations=100):
    alm_penalty_scalar_rho = 1.5
    tolerance = 1e-10
    err = []
    num_pixels_n, num_frames_p = frames_D.shape
    regularization_lambda = 1.0 / np.sqrt(num_pixels_n)
    dual_mu = 12.5 / la.norm(frames_D, ord=2)
    dual_Y = frames_D / dual_norm(frames_D, regularization_lambda)
    foreground_S = np.zeros_like(frames_D) # E in reference code
    background_L = np.zeros_like(frames_D) # A in reference code
    
    rk = min(num_pixels_n, num_frames_p)
    for t in range(max_iterations):
        background_L, rk = _calc_background_L(frames_D, dual_Y, dual_mu, foreground_S, rk)
        foreground_S = _calc_foreground_S(frames_D, dual_Y, dual_mu, background_L, graph, regularization_lambda)
        dual_Y = _calc_Y(frames_D, dual_Y, dual_mu, background_L, foreground_S)
        dual_mu = alm_penalty_scalar_rho * dual_mu
        err.append(_calc_error(frames_D, background_L, foreground_S))
        if err[-1] < tolerance:
            break
    return background_L, foreground_S, err


# block spase RPCA
def _calc_foreground_S_bs(frames_D, dual_Y, dual_mu, background_L, group_info):
    G = frames_D - background_L + dual_Y / dual_mu
    ret_S = np.zeros_like(G)
    for i in range(len(group_info)):
        loc_i = group_info[i]['index_i']
        loc_j = group_info[i]['index_j']
        thresh = group_info[i]['lambda'] / dual_mu
        
        val = la.norm(G[loc_i, loc_j])
        
        coeff = 0;
        if(val > thresh): coeff = (val - thresh)/val;
        ret_S[loc_i, loc_j] = coeff * G[loc_i, loc_j]
        
    return ret_S

def inexact_alm_bs(frames_D, group_info, max_iterations=100):
    alm_penalty_scalar_rho = 1.5
    tolerance = 1e-7
    err = []    
    num_pixels_n, num_frames_p = frames_D.shape
    ref_lambda = 1.0 / np.sqrt(num_pixels_n)
    mu0 = 12.5 / la.norm(frames_D, ord=2)
    
    dual_mu = mu0
    dual_Y = frames_D / dual_norm(frames_D, ref_lambda)
    foreground_S = np.zeros_like(frames_D) # E in reference code
    background_L = np.zeros_like(frames_D) # A in reference code
    
    for t in range(max_iterations):
        background_L = _calc_background_L(frames_D, dual_Y, dual_mu, foreground_S)
        foreground_S = _calc_foreground_S_bs(frames_D, dual_Y, dual_mu, background_L, group_info)
        dual_Y = _calc_Y(frames_D, dual_Y, dual_mu, background_L, foreground_S)
        dual_mu = min(alm_penalty_scalar_rho * dual_mu, tolerance * mu0)
        err.append(_calc_error(frames_D, background_L, foreground_S))
        if err[-1] < tolerance:
            break
    return background_L, foreground_S, err
