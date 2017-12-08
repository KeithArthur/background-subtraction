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
import motion as m
import group
from alm_lsd import inexact_alm_lsd, inexact_alm_bs

from FRPCA_GD import FRPCA
import cPickle as pickle
import os.path
import sys

project_path = '../'
video_name = ''
sys.path.append(project_path + 'src/')

def read_images(data_name):
    import glob
    import re
    
    def numericalSort(value):
        _numbers = re.compile(r'(\d+)')
        parts = _numbers.split(value)
        parts[1::2] = map(int, parts[1::2])
        return parts

    frame_path = project_path + 'data/' + data_name + '/'
    frames = []
    file_list = glob.glob(frame_path + "*.jpg")
    for file_names in sorted(file_list, key=numericalSort):
        image = PIL.Image.open(file_names)
        frame = np.asarray(image)
        frames.append(frame)
        
    return [0, len(frames)], np.array(frames)

def first_pass(frames_D, original_mean, frame_dimensions, downsampled_frame_dimensions, num_frames, norm_choice):
    if norm_choice == 'l1':
        print ('L1')
        background_L, foreground_S, err = inexact_alm_lsd(frames_D, None, 'l1')
        dims = downsampled_frame_dimensions
    elif norm_choice == 'structured_sparsity':
        print ('Structured Sparsity')
        batch_dimensions = [3, 3]
        graph = g.build_graph(downsampled_frame_dimensions, batch_dimensions)
        background_L, foreground_S, err = inexact_alm_lsd(frames_D, graph, 'structured_sparsity')
        dims = downsampled_frame_dimensions
    elif norm_choice == 'frpca':
        print ('FRPCA')
        background_L, foreground_S, err = FRPCA(frames_D, alpha = .3, r=1)
        dims = frame_dimensions
    else:
        raise ValueError('norm_choice is not valid.')

    # Masking first-RPCA foreground & Upsampling
    masked_S = f.foreground_mask(np.abs(foreground_S), frames_D, background_L)
    fg_frames = f.matrix_to_frames(masked_S, num_frames, dims)
    upsampled_fg = f.resize_frames(fg_frames, frame_dimensions)
    upsampled_fg = np.int32(upsampled_fg > 128) * 255

    # print
    bg_bin = f.restore_background(f.matrix_to_frames(background_L, num_frames,
                                                     dims), original_mean)
    bg_images = [PIL.Image.fromarray(frame) for frame in bg_bin]
    fg_images = [PIL.Image.fromarray(frame) for frame in upsampled_fg]
    for i in range(len(fg_images)):
        fg_images[i].save(project_path + "src/foreground_first_pass/out" + str(i) + norm_choice + ".gif")
        bg_images[i].save(project_path + "src/background_first_pass/out" + str(i) + norm_choice + ".gif")
    return upsampled_fg


def main():
    frame_index = [0, 48]
    video_data_path = project_path + 'LSD/data/WaterSurface.mat'
    all_frames = scipy.io.loadmat(video_data_path)['ImData']
    start_frame_index = frame_index[0]
    end_frame_index = frame_index[1]
    frames_to_process = np.rollaxis(all_frames[:, :, start_frame_index:end_frame_index], 2)

    if video_name is not '':
        frame_index, frames_to_process = read_images(video_name)
    frame_dimensions = frames_to_process.shape[1:]

    downsampling_ratio = 1.0 / 4.0
    downsampled_frames = f.resize_frames(frames_to_process, downsampling_ratio)
    downsampled_frame_dimensions = downsampled_frames.shape[1:]

    num_frames = downsampled_frames.shape[0]
    normalized_frames, original_mean = f.normalize_and_center_frames(downsampled_frames)

    frames_D = f.frames_to_matrix(normalized_frames, num_frames, downsampled_frame_dimensions)

    print ('---Identifying Trajectories---')
    if os.path.isfile('save.p'):
        trajectories = pickle.load(open( "save.p", "rb" ))
    else:
        # find trajectory
        optical_flows = m.calc_forward_backward_flow(frames_to_process)
        trajectories = m.calc_trajectories(optical_flows[0], optical_flows[1], frame_dimensions, 5)
        pickle.dump(trajectories, open( "save.p", "wb" ))

    for norm_choice in ['l1', 'structured_sparsity', 'frpca']:
        upsampled_fg = first_pass(frames_D, original_mean, frame_dimensions, downsampled_frame_dimensions, num_frames, norm_choice)

        print ('---Calculating Group Regularization Parameters---')
        # identify ground and compute lambda
        video_data_dimensions = [num_frames] + list(frame_dimensions)
        groups_info = group.find_groups(upsampled_fg, num_frames, upsampled_fg.shape[1:], min_size=50)
        m.set_groups_saliencies(groups_info, trajectories, video_data_dimensions, 5)
        m.set_regularization_lambdas(groups_info, video_data_dimensions)

        print ('---Group Sparse RPCA---')
        # group sparse RPCA
        normalized_frames, original_mean = f.normalize_and_center_frames(frames_to_process)
        frames_D = f.frames_to_matrix(normalized_frames, num_frames, frame_dimensions)
        final_L, final_S, err = inexact_alm_bs(frames_D, groups_info)

        # finalize S, L
        final_bg = f.restore_background(f.matrix_to_frames(final_L, num_frames, frame_dimensions), original_mean)
        masked_fg = f.foreground_mask(np.abs(final_S), frames_D, final_L)
        final_fg = f.matrix_to_frames(masked_fg, num_frames, frame_dimensions)

        # print
        fg_images = [PIL.Image.fromarray(frame) for frame in final_fg]
        bg_images = [PIL.Image.fromarray(frame) for frame in final_bg]

        for i in range(len(fg_images)):
            fg_images[i].save(project_path + "src/foreground/out" + str(i) + norm_choice + ".gif")
            bg_images[i].save(project_path + "src/background/out" + str(i) + norm_choice + ".gif")

        bg_images[0].save(project_path + norm_choice + "bg_out.gif", save_all=True, append_images=bg_images[1:])
        fg_images[0].save(project_path + norm_choice + "fg_out.gif", save_all=True, append_images=fg_images[1:])
    

if __name__ == "__main__":
    main()
