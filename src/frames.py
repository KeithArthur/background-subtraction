import numpy as np
import scipy

def resize_frames(frames, ratio):
    return np.array([scipy.misc.imresize(frames[..., frame_index], ratio) for frame_index in range(frames.shape[-1])])

def _normalize_frame(frame):
    return np.float32(frame) / np.max(frame)

def _normalize_frames(frames):
    return np.array([_normalize_frame(frame) for frame in frames])

def normalize_and_center_frames(frames):
    normalized_frames = _normalize_frames(frames)
    mean = np.mean(normalized_frames)
    return normalized_frames - mean

def restore_frames(frames, original_mean):
    return np.int32(256 * _normalize_frames(frames + original_mean))
