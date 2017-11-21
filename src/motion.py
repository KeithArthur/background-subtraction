import numpy as np
import numpy.linalg as la
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
    position_indices = [np.flip(pos, 0) for pos in trajectories['positions']]
    for row, col in _get_remaining_indices(position_indices, flow.shape[:2]):
        delta = flow[row, col]
        trajectories['deltas'].append(left_pad([np.array(delta)], np.nan, frames_so_far))
        trajectories['positions'].append(np.array([col, row]) + np.floor(delta))

def _update_trajectories(flow, trajectories):
    for index, pos in enumerate(trajectories['positions']):
        delta = flow[int(pos[1]), int(pos[0])]
        trajectories['positions'][index] += np.floor(delta)
        trajectories['deltas'][index].append(delta)

def _flows_close(forward, backward):
    return la.norm(forward + backward)**2 < 0.01 * la.norm(forward)**2 + la.norm(backward)**2 + 0.2

def _end_occluded_trajectories(forward_flow, backward_flow, trajectories):
    complete_trajectories = {'positions': [], 'deltas': []}
    for index, pos in enumerate(trajectories['positions']):
        col, row = np.int32(pos)
        if _flows_close(forward_flow[row, col], backward_flow[row, col]):
            complete_trajectories['positions'].append(trajectories['positions'].pop(index))
            complete_trajectories['deltas'].append(trajectories['deltas'].pop(index))
    return complete_trajectories

def calc_trajectories(forward_flows, backward_flows):
    trajectories = {'positions': [], 'deltas': []}
    for forward_flow, backward_flow in zip(forward_flows, backward_flows):
        _update_trajectories(forward_flow, trajectories)
        _init_missing_trajectories(forward_flow, trajectories)

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
