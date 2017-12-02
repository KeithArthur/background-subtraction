import numpy as np
import numpy.linalg as la
import cv2

from utils import prev_cur_next, left_pad, extend_dict, enumerate_pairs_with_order
import group as g

import platform
project_float = np.float64 if '64' in platform.architecture()[0] else np.float32


def _get_remaining_indices(trajectories_current_positions, shape):
    all_indices = np.indices(shape).reshape(2, np.prod(shape)).T.astype(np.int32).tolist()
    positions = {tuple(coll.tolist()) for coll in trajectories_current_positions}
    return [index for index in all_indices if tuple(index) not in positions]

def _init_missing_trajectories(flow, trajectories):
    frames_so_far = len(trajectories['deltas'][0]) if len(trajectories['deltas']) != 0 else 0
    position_indices = [np.flip(pos, 0) for pos in trajectories['positions']]
    for row, col in _get_remaining_indices(position_indices, flow.shape[:2]):
        delta = flow[row, col]
        trajectories['deltas'].append(left_pad([], np.nan, frames_so_far))
        trajectories['positions'].append(np.array([col, row], dtype=project_float))

def _update_trajectories(flow, trajectories, frame_dimensions):
    for index, pos in enumerate(trajectories['positions']):
        delta = flow[int(pos[1]), int(pos[0])]
        trajectories['positions'][index] = np.clip(trajectories['positions'][index] + np.floor(delta),
                                                   [0, 0],
                                                   np.flip(np.array(frame_dimensions), 0) - 1)
        trajectories['deltas'][index].append(delta)

def _flows_close(forward, backward):
    return np.sum((forward + backward)**2) < 0.01 * (np.sum(forward**2) + np.sum(backward**2)) + 0.2

def _end_occluded_trajectories(forward_flow, backward_flow, trajectories):
    complete_trajectories = {'positions': [], 'deltas': []}
    for index, pos in enumerate(trajectories['positions']):
        col, row = np.int32(pos)
        if _flows_close(forward_flow[row, col], backward_flow[row, col]):
            complete_trajectories['positions'].append(trajectories['positions'].pop(index))
            complete_trajectories['deltas'].append(trajectories['deltas'].pop(index))
    return complete_trajectories

def _is_salient(trajectory_deltas):
    positive_motion_P = {'horiz': 0, 'vert': 0}
    negative_motion_N = {'horiz': 0, 'vert': 0}
    without_nans = [delta for delta in trajectory_deltas if not np.isscalar(delta)]
    for delta in without_nans:
        if delta[0] > 0:
            positive_motion_P['horiz'] += 1
        elif delta[0] < 0:
            negative_motion_N['horiz'] += 1
        if delta[1] > 0:
            positive_motion_P['vert'] += 1
        elif delta[1] < 0:
            negative_motion_N['vert'] += 1
    return any(np.add(positive_motion_P.values(), negative_motion_N.values()) > 0.8 * len(without_nans))

def _get_inconsistent_trajectory_nums(trajectories):
    trajectory_nums = []
    for index, deltas in enumerate(trajectories['deltas']):
        if _is_salient(deltas):
            continue
        else:
            trajectory_nums.append(index)
    return trajectory_nums

def _deltas_to_positions(trajectories):
    positions = []
    for index, position in enumerate(trajectories['positions']):
        trajectory_positions = [position]
        without_nans = [delta for delta in trajectories['deltas'][index] if not np.isscalar(delta)]
        for delta in reversed(without_nans):
            trajectory_positions.append(trajectory_positions[-1] - np.floor(delta))
        positions.append(list(reversed(trajectory_positions)))
    return positions

def _calc_trajectory_saliencies(trajectories):
    positions = _deltas_to_positions(trajectories)
    saliencies = []
    inconsistent_trajectory_nums = _get_inconsistent_trajectory_nums(trajectories)
    for trajectory_num, trajectory_positions in enumerate(positions):
        if trajectory_num in inconsistent_trajectory_nums:
            saliencies.append(0)
        else:
            saliencies.append(np.max([la.norm(position_1 - position_2) for position_1, position_2 in enumerate_pairs_with_order(trajectory_positions)]))
    return saliencies

def _get_pixel_trajectory_lookup(trajectories, video_data_dimensions):
    trajectory_positions = _deltas_to_positions(trajectories)
    pixel_trajectory_lookup = np.ones(video_data_dimensions, dtype=np.int32) * -1
    for trajectory_num, trajectory in enumerate(trajectory_positions):
        for index, position in enumerate(trajectory):
            row = int(position[1])
            col = int(position[0])
            pixel_trajectory_lookup[index, row, col] = trajectory_num
    return pixel_trajectory_lookup

def _get_pixel_saliencies(trajectory_saliencies, pixel_trajectory_lookup):
    non_salient_value = 0.0
    pixel_saliencies = np.zeros_like(pixel_trajectory_lookup, dtype=project_float)
    for index, trajectory_num in np.ndenumerate(pixel_trajectory_lookup):
        if trajectory_num >= 0:
            pixel_saliencies[index] = trajectory_saliencies[trajectory_num]
        else:
            pixel_saliencies[index] = non_salient_value
    return pixel_saliencies

def calc_forward_backward_flow(frames):
    forward_flow = []
    backward_flow = []
    for prev_frame, cur_frame, next_frame in prev_cur_next(frames):
        if prev_frame is not None:
            backward_flow.append(cv2.calcOpticalFlowFarneback(prev_frame, cur_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0))
        if next_frame is not None:
            forward_flow.append(cv2.calcOpticalFlowFarneback(cur_frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0))
    return (forward_flow, backward_flow)

def calc_trajectories(forward_flows, backward_flows, frame_dimensions):
    trajectories = {'positions': [], 'deltas': []}
    completed_trajectories = {'positions': [], 'deltas': []}
    for forward_flow, backward_flow in zip(forward_flows, backward_flows):
        _init_missing_trajectories(forward_flow, trajectories)
        _update_trajectories(forward_flow, trajectories, frame_dimensions)
        extend_dict(completed_trajectories, _end_occluded_trajectories(forward_flow, backward_flow, trajectories))
    extend_dict(completed_trajectories, trajectories)
    return completed_trajectories

def set_groups_saliencies(groups, trajectories, video_data_dimensions):
    pixel_trajectory_lookup = _get_pixel_trajectory_lookup(trajectories, video_data_dimensions)
    trajectory_saliencies = _calc_trajectory_saliencies(trajectories)
    pixel_saliencies = _get_pixel_saliencies(trajectory_saliencies, pixel_trajectory_lookup)
    for group in groups:
        group_pixel_saliencies = g.keep_only_in_group(pixel_saliencies[group['frame']], group['elems'])
        group['salience'] = np.sum(group_pixel_saliencies) / len(group['elems'])
    return groups

def set_regularization_lambdas(groups, video_data_dimensions):
    min_salience = min([group['salience'] for group in groups])
    normalization = min_salience / np.sqrt(np.max(video_data_dimensions[1:]))
    for group in groups:
        if group['salience'] == 0:
            group['regularization_lambda'] = 10.0
        else:
            group['regularization_lambda'] = 0.1 * normalization / group['salience']
    return groups
