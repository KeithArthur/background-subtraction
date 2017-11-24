import numpy as np
import scipy

import platform
project_float = np.float64 if '64' in platform.architecture()[0] else np.float32

def from_video_data(video_data):
    return np.rollaxis(video_data, 2)

def to_video_data(frames):
    return np.rollaxis(frames, 0, 3)

def to_flattened_frames(frames, frame_dimensions, num_frames):
    return to_video_data(frames).reshape(np.prod(frame_dimensions), num_frames)

def from_flattened_frames(flattend_frames, video_data_shape):
    return np.rollaxis(flattend_frames.reshape(video_data_shape), 2)

def resize_frames(frames, ratio):
    return np.array([scipy.misc.imresize(frame, ratio) for frame in frames])

def _normalize_frame(frame):
    return project_float(frame) / np.max(frame)

def _normalize_frames(frames):
    return np.array([_normalize_frame(frame) for frame in frames])

def normalize_and_center_frames(frames):
    normalized_frames = _normalize_frames(frames)
    mean = np.mean(normalized_frames)
    return normalized_frames - mean

def restore_frames(flattend_frames, original_frames):
    normalized_mean = np.mean(_normalize_frames(original_frames))
    video_data_shape = to_video_data(original_frames).shape
    frames = from_flattened_frames(flattend_frames, video_data_shape)
    return np.int32(256 * (_normalize_frames(frames) + normalized_mean))
