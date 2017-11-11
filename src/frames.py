import numpy as np
import scipy

def resize_frames(frames, ratio):
    return np.array([scipy.misc.imresize(frame, ratio) for frame in frames])

def _normalize_frame(frame):
    return frame / np.max(frame)

def _center_frames(frames):
    return frames - np.mean(frames)

def normalize_and_center_frames(frames):
    return _center_frames(np.array([_normalize_frame(frame) for frame in frames]))
