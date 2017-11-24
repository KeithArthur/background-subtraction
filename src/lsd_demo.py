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
<<<<<<< HEAD
    end_frame_index = frame_index[1]
    frames_to_process = np.rollaxis(all_frames[:, :, start_frame_index:end_frame_index], 2)
    
=======
    end_frame_index = frame_index[1] + 1
    frames_to_process = f.from_video_data(video_data[:, :, start_frame_index:end_frame_index])

>>>>>>> da1df28431c54b166cdb82d5941a9ae65dafda9c
    downsampling_ratio = 1.0 / 4.0
    downsampled_frames = f.resize_frames(frames_to_process, downsampling_ratio)
    frame_dimensions = downsampled_frames.shape[1:]
    
    #PIL.Image.fromarray(frames_to_process[35]).show() -> checked roll axis
    num_frames = downsampled_frames.shape[0]
<<<<<<< HEAD
    normalized_frames, original_mean = f.normalize_and_center_frames(downsampled_frames)
    
    # 48 X (image) -> 48 X ( Flattened vector ) -> Transpose
    frames_D = normalized_frames.transpose(0,2,1).reshape(num_frames, np.prod(frame_dimensions)).T
=======
    normalized_frames = f.normalize_and_center_frames(downsampled_frames)
    frames_D = f.to_flattened_frames(normalized_frames, frame_dimensions, num_frames)
>>>>>>> da1df28431c54b166cdb82d5941a9ae65dafda9c
    batch_dimensions = [3, 3]
    graph = g.build_graph(frame_dimensions, batch_dimensions)
    background_L, foreground_S, err = inexact_alm_lsd(frames_D, graph)
<<<<<<< HEAD
    bin_bg = f.restore_frames(background_L.T.reshape(num_frames,frame_dimensions[1],frame_dimensions[0]).transpose(0,2,1), original_mean)
    
    # Mask function needed
    masked_S = f.foreground_mask(np.abs(foreground_S), frames_D, background_L)
    bin_fg = masked_S.T.reshape(num_frames,frame_dimensions[1],frame_dimensions[0]).transpose(0,2,1)
=======

    bg = f.restore_frames(background_L, downsampled_frames)
    fg = f.restore_frames(foreground_S, downsampled_frames)
    bin_fg = np.maximum(fg, 0)
>>>>>>> da1df28431c54b166cdb82d5941a9ae65dafda9c
    fg_images = [PIL.Image.fromarray(frame) for frame in bin_fg]
    bg_images = [PIL.Image.fromarray(frame) for frame in bin_bg]

    for i in range(len(fg_images)):
        fg_images[i].save("./foreground/out" + str(i) + ".gif")
        bg_images[i].save("./background/out" + str(i) + ".gif")
    

if __name__ == "__main__":
    main()
