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
import glob
import re

def numericalSort(value):
    _numbers = re.compile(r'(\d+)')
    parts = _numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def read_images(data_name):
    frame_path = '../data/' + data_name + '/'
    frames = []
    file_list = glob.glob(frame_path + "*.jpg")
    for file_names in sorted(file_list, key=numericalSort):
        image = PIL.Image.open(file_names)
        frame = np.asarray(image)
        frames.append(frame)
        
    return [0, len(frames)], np.array(frames)
    
    
def main():
    """
    frame_index = [0, 48]
    video_data_path = '../LSD/data/WaterSurface.mat'
    all_frames = scipy.io.loadmat(video_data_path)['ImData']
    start_frame_index = frame_index[0]
    end_frame_index = frame_index[1]
    frames_to_process = np.rollaxis(all_frames[:, :, start_frame_index:end_frame_index], 2)
    """
    
    frame_index, frames_to_process = read_images('boats')
    
    downsampling_ratio = 1.0 / 4.0
    downsampled_frames = f.resize_frames(frames_to_process, downsampling_ratio)
    frame_dimensions = downsampled_frames.shape[1:]
    
    #PIL.Image.fromarray(frames_to_process[35]).show() -> checked roll axis
    num_frames = downsampled_frames.shape[0]
    normalized_frames, original_mean = f.normalize_and_center_frames(downsampled_frames)
    
    frames_D = f.frames_to_matrix(normalized_frames, num_frames, frame_dimensions)
    batch_dimensions = [3, 3]
    graph = g.build_graph(frame_dimensions, batch_dimensions)
    background_L, foreground_S, err = inexact_alm_lsd(frames_D, graph)
    bin_bg = f.restore_background(f.matrix_to_frames(background_L, num_frames, frame_dimensions), original_mean)
    
    # Mask function needed
    masked_S = f.foreground_mask(np.abs(foreground_S), frames_D, background_L)
    bin_fg = f.matrix_to_frames(masked_S, num_frames, frame_dimensions)
    fg_images = [PIL.Image.fromarray(frame) for frame in bin_fg]
    bg_images = [PIL.Image.fromarray(frame) for frame in bin_bg]

    for i in range(len(fg_images)):
        fg_images[i].save("./foreground/out" + str(i) + ".gif")
        bg_images[i].save("./background/out" + str(i) + ".gif")
    

if __name__ == "__main__":
    main()
