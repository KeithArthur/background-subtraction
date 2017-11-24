from __future__ import print_function
import PIL.Image
import scipy
import scipy.sparse as sp
import scipy.io
import numpy as np
import numpy.linalg as la
import numpy.random as rn
import matplotlib.pyplot as plt

import frames as f
import graph as g
from alm_lsd import inexact_alm_lsd


def main():
    frame_index = [0, 48]
    video_data_path = '../LSD/data/WaterSurface.mat'
    video_data = scipy.io.loadmat(video_data_path)['ImData']
    start_frame_index = frame_index[0]
    end_frame_index = frame_index[1] + 1
    frames_to_process = f.from_video_data(video_data[:, :, start_frame_index:end_frame_index])

    downsampling_ratio = 1.0 / 4.0
    downsampled_frames = f.resize_frames(frames_to_process, downsampling_ratio)
    frame_dimensions = downsampled_frames.shape[1:]
    num_frames = downsampled_frames.shape[0]
    normalized_frames = f.normalize_and_center_frames(downsampled_frames)
    frames_D = f.to_flattened_frames(normalized_frames, frame_dimensions, num_frames)
    batch_dimensions = [3, 3]
    graph = g.build_graph(frame_dimensions, batch_dimensions)

    background_L, foreground_S, err = inexact_alm_lsd(frames_D, graph)

    bg = f.restore_frames(background_L, downsampled_frames)
    fg = f.restore_frames(foreground_S, downsampled_frames)
    bin_fg = np.maximum(fg, 0)
    fg_images = [PIL.Image.fromarray(frame) for frame in bin_fg]
    fg_images[0].save("out.gif", save_all=True, append_images=fg_images[1:])


if __name__ == "__main__":
    main()
