import numpy as np
import scipy

import platform
project_float = np.float64 if '64' in platform.architecture()[0] else np.float32

def resize_frames(frames, ratio):
    return np.array([scipy.misc.imresize(frame, ratio) for frame in frames])

#Normalize should be done uniformly, otherwise it will lose low-rank property after mean subtraction.
def _normalize_frame(frame):
    return frame / 255.0
    #return project_float(frame) / np.max(frame)

def _normalize_frames(frames):
    return frames / 255.0
    #return np.array([_normalize_frame(frame) for frame in frames])

def normalize_and_center_frames(frames):
    normalized_frames = _normalize_frames(frames)
    mean = np.mean(normalized_frames)
    return normalized_frames - mean, mean

# 48 X (image) -> 48 X ( Flattened vector ) -> Transpose, Column first counting
def frames_to_matrix(M, frame_n, frame_d):
    return M.transpose(0,2,1).reshape(frame_n, np.prod(frame_d)).T
    
def matrix_to_frames(M, frame_n, frame_d):
    return M.T.reshape(frame_n,frame_d[1],frame_d[0]).transpose(0,2,1)

def restore_background(frames, original_mean):
    return np.int32(255.0 * (frames + original_mean))

def foreground_mask(S, M, L):
    S_back_temp = (S < (0.5 * np.max(S)))
    S_diff = np.array(list((np.abs(M-L) * S_back_temp).flat))
    
    S_sel = S_diff[S_diff>0]
    mu_s = np.mean(S_sel)
    sigma_s = np.std(S_sel)
    th = mu_s + 1.5*sigma_s
    Mask = S > th
    return np.int32(Mask) * 255