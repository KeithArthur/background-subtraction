import numpy as np
import cv2

from utils import prev_cur_next, left_pad

def calc_flow(frames):
    flow = []
    for prev_frame, cur_frame, next_frame in prev_cur_next(frames):
        if next_frame is None:
            break
        flow.append(cv2.calcOpticalFlowFarneback(cur_frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0))
    return np.array(flow)


def calc_forward_backward_flow(frames):
    forward_flow = []
    backward_flow = []
    for prev_frame, cur_frame, next_frame in prev_cur_next(frames):
        if prev_frame != None:
            backward_flow.append(cv2.calcOpticalFlowFarneback(prev_frame, cur_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0))
        if next_frame != None:
            forward_flow.append(cv2.calcOpticalFlowFarneback(cur_frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0))
    return np.array(forward_flow, backward_flow)

def _get_remaining_indices(trajectories_current_positions, shape):
    all_indices = np.indices(shape).reshape(2, np.prod(shape)).T.astype(np.int32).tolist()
    positions = [coll.tolist() for coll in trajectories_current_positions]
    return [index for index in all_indices if index not in positions]

def _init_missing_trajectories(flow, trajectories):
    frames_so_far = len(trajectories['deltas'])
    for row, col in _get_remaining_indices(trajectories['positions'], flow.shape[:2]):
        delta = flow[row, col]
        trajectories['deltas'].append([left_pad(np.array(delta), [np.nan, np.nan], frames_so_far)])
        trajectories['positions'].append([np.array([row, col]) + np.floor(delta)])

def flow_into_trajectories(flow, trajectories):
    for index, pos in enumerate(trajectories['positions']):
        delta = flow[pos['y'], pos['x']]
        pos += np.floor(delta)
        trajectories['deltas'][index].append(delta)

def calc_trajectories(forward_flow, backward_flow):
    trajectories = None
    for flow, back in zip(forward_flow, backward_flow):
        flow = cv2.calcOpticalFlowFarneback(cur_frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        if trajectories is None:
            trajectories = np.array([flow])

# # params for ShiTomasi corner detection
# feature_params = dict( maxCorners = 100,
#                        qualityLevel = 0.3,
#                        minDistance = 7,
#                        blockSize = 7 )
# # Parameters for lucas kanade optical flow
# lk_params = dict( winSize  = (15,15),
#                   maxLevel = 2,
#                   criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# old_frame = all_frames[..., 0]
# old_gray = old_frame
# p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# i = 1
# while(1):
#     frame = all_frames[..., i]
#     frame_gray = frame
#     # calculate optical flow
#     p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
#     # Select good points
#     good_new = p1[st==1]
#     good_old = p0[st==1]
#     # draw the tracks
#     # Now update the previous frame and previous points
#     old_gray = frame_gray.copy()
#     p0 = good_new.reshape(-1,1,2)
# cv2.destroyAllWindows()
