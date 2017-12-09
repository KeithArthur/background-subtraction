import numpy as np
import numpy.linalg as la
import cv2

from utils import prev_cur_next, left_pad, extend_dict, enumerate_pairs_with_order
import group as g

import platform
project_float = np.float64 if '64' in platform.architecture()[0] else np.float32


def _get_remaining_indices(trajectories_current_positions, flow, shape):
    Hi = np.abs(flow[:,:,0]) + np.abs(flow[:,:,1])
    for index, pos in enumerate(trajectories_current_positions):
        x, y = pos[-1][0], pos[-1][1]
        Hi[x,y] = 0
    
    return zip(*np.where(Hi >= 0.5))

def _init_missing_trajectories(flow, trajectories, frames_so_far):
    for row, col in _get_remaining_indices(trajectories['positions'], flow, flow.shape[:2]):
        trajectories['st_frame'].append(frames_so_far)
        trajectories['deltas'].append([])
        trajectories['positions'].append([np.array([row, col])])

def _flows_close(forward, backward):
    return np.sum((forward + backward)**2) < 0.01 * (np.sum(forward**2) + np.sum(backward**2)) + 0.2

def _update_trajectories(flow_f, flow_b, trajectories, complete_trajectories, frame_dimensions, len_thresh):
    frd = np.array(frame_dimensions) - 1    
    for index, pos in enumerate(trajectories['positions']):
        delta = flow_f[pos[-1][0], pos[-1][1]]
        n_pos = np.clip(pos[-1] + delta, [0,0], frd)
        nx, ny = int(n_pos[0]), int(n_pos[1])
        ff, bb = flow_f[nx, ny], flow_b[nx, ny]
        
        if _flows_close(ff, bb):
            tp = trajectories['positions'].pop(index)
            td = trajectories['deltas'].pop(index)
            tf = trajectories['st_frame'].pop(index)
            
            if( len(td) > len_thresh and _is_salient(td)):
                complete_trajectories['positions'].append(tp)
                complete_trajectories['st_frame'].append(tf)
                
        else:            
            trajectories['positions'][index].append(np.array([nx, ny]))
            trajectories['deltas'][index].append(delta)


def _is_salient(trajectory_deltas):
    positive_motion_P = {'horiz': 0, 'vert': 0}
    negative_motion_N = {'horiz': 0, 'vert': 0}

    for delta in trajectory_deltas:
        if delta[1] > 0.1:
            positive_motion_P['vert'] += 1
        elif delta[1] < -0.1:
            negative_motion_N['vert'] += 1
        if delta[0] > 0.1:
            positive_motion_P['horiz'] += 1
        elif delta[0] < -0.1:
            negative_motion_N['horiz'] += 1
    return any(np.add(positive_motion_P.values(), negative_motion_N.values()) > 0.5 * len(trajectory_deltas))

def _calc_trajectory_saliencies(trajectories, len_thresh):
    positions = trajectories['positions']
    saliencies = []
    
    for trajectory_num, trajectory_positions in enumerate(positions):
        if( len(trajectory_positions) < len_thresh ):
            saliencies.append(0)
        else:
            saliencies.append(np.max([la.norm(position_1 - position_2) for position_1, position_2 in enumerate_pairs_with_order(trajectory_positions)]))
        
    return saliencies

def _get_pixel_trajectory_lookup(trajectories, video_data_dimensions):
    trajectory_positions = trajectories['positions']
    pixel_trajectory_lookup = np.ones(video_data_dimensions, dtype=np.int32) * -1
    
    for trajectory_num, trajectory in enumerate(trajectory_positions):
        f_num = trajectories['st_frame'][trajectory_num]
        for index, position in enumerate(trajectory):
            row = int(position[0])
            col = int(position[1])
            pixel_trajectory_lookup[f_num + index, row, col] = trajectory_num
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
            
    #print forward_flow[2][5,6], np.flip(forward_flow, 3)[2][5,6]
    return (np.flip(forward_flow, 3), np.flip(backward_flow, 3))

def calc_trajectories(forward_flows, backward_flows, frame_dimensions, len_thresh):
    f_num = 0
    trajectories = {'positions': [], 'deltas': [], 'st_frame': []}
    completed_trajectories = {'positions': [], 'st_frame': []}
    
    for forward_flow, backward_flow in zip(forward_flows, backward_flows):
        _init_missing_trajectories(forward_flow, trajectories, f_num)
        _update_trajectories(forward_flow, backward_flow, trajectories, completed_trajectories, frame_dimensions, len_thresh)
        f_num += 1

    extend_dict(completed_trajectories, trajectories)       
    return completed_trajectories

def set_groups_saliencies(groups, trajectories_f, trajectories_b, video_data_dimensions):
    pixel_trajectory_lookup = _get_pixel_trajectory_lookup(trajectories_f, video_data_dimensions)
    trajectory_saliencies_f = _calc_trajectory_saliencies(trajectories_f, 5)
    pixel_saliencies_f = _get_pixel_saliencies(trajectory_saliencies_f, pixel_trajectory_lookup)
    
    pixel_trajectory_lookup = _get_pixel_trajectory_lookup(trajectories_b, video_data_dimensions)
    trajectory_saliencies_b = _calc_trajectory_saliencies(trajectories_b, 5)
    pixel_saliencies_b = _get_pixel_saliencies(trajectory_saliencies_b, pixel_trajectory_lookup)
    
    pixel_saliencies = (pixel_saliencies_f + np.flip(pixel_saliencies_b, 0)) / 2.0
        
    import PIL.Image
    sal_map = [PIL.Image.fromarray(10 * frame) for frame in pixel_saliencies]
    for i in range(len(sal_map)):
        sal_map[i].save("./sal_map/out" + str(i) + ".gif")
        
    for group in groups:
        group_pixel_saliencies = g.keep_only_in_group(pixel_saliencies[group['frame']], group['elems'])
        group['salience'] = np.sum(group_pixel_saliencies) / len(group['index'])
        
    return groups
    
def set_regularization_lambdas(groups, video_data_dimensions):
    min_salience = 10000000
    for group in groups:
        if(group['salience'] > 1 and min_salience > group['salience'] ):
            min_salience = group['salience']
    
    normalization = min_salience / np.sqrt(np.max(video_data_dimensions[1:]))
    for group in groups:
        if group['salience'] <= 1:
            group['regularization_lambda'] = 1000.0
        else:
            group['regularization_lambda'] = normalization / group['salience'] / len(group['index'])
    return groups

"""
def set_groups_saliencies_from_flows(groups, flows):
    def flow_mag(flows):
        f = []
        for flow in flows:
            arr = flow.reshape(np.prod(flow.shape[:2]), 2)
            f.append(np.array([la.norm(e) for e in arr]).reshape(flow.shape[:2]))
        return f
    mag = flow_mag(flows)
    pixel_saliencies = mag / np.max(mag) * 1000.0
    for group in groups:
        if group['frame'] > 29:
            group['salience'] = 0.0
            continue
        group_pixel_saliencies = g.keep_only_in_group(pixel_saliencies[group['frame']], group['elems'])
        group['salience'] = np.sum(group_pixel_saliencies) / len(group['elems'])
    return groups, pixel_saliencies
"""