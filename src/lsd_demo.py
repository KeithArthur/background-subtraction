from __future__ import print_function
import scipy
import scipy.sparse as sp
import numpy as np
import numpy.linalg as la
import numpy.random as rn
import matplotlib.pyplot as plt

import spams

import frames as f
import graph as g
from alm_lsd import inexact_alm_lsd


def main():
    frame_index = [1, 48]
    video_data_path = '../LSD/data/WaterSurface.mat'
    all_frames = scipy.io.loadmat(video_data_path)
    start_frame_index = frame_index[0]
    end_frame_index = frame_index[1]
    frames_to_process = all_frames[:, :, start_frame_index, end_frame_index]
    M, N, T = frames_to_process.shape

    downsampling_ratio = 1.0 / 4.0
    downsampled_frames = f.resize_frames(frames_to_process, downsampling_ratio)
    frame_dimensions = downsampled_frames.shape[:2]
    num_frames = downsampled_frames.shape[2]
    normalized_frames = f.normalize_and_center_frames(downsampled_frames)
    frames_D = normalized_frames.reshape(np.prod(frame_dimensions), num_frames)
    batch_dimensions = [3, 3]
    graph = g.build_graph(frame_dimensions, batch_dimensions)

    background_L, foreground_S, num_iterations = inexact_alm_lsd(frames_D, graph)
