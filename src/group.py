# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 17:12:37 2017

@author: Bisu
"""

import numpy as np
import utils

def check_inside_bound(x, y, f_dim):
    return x >= 0 and y >= 0 and x < f_dim[0] and y < f_dim[1]
    
def find_groups(fg_images, num_frames, f_dim):
    groups = []
    search_dir = [[+1, 0], [0, +1], [-1, 0], [0, -1]]
    
    for frame_num in range(num_frames):
        have_visited = np.zeros(f_dim)
        for x0 in range(f_dim[0]):
            for y0 in range(f_dim[1]):
                if( fg_images[frame_num][x0, y0] <= 0 or have_visited[x0, y0] == 1 ): 
                    continue
                
                gr_list, ind_list, queue = [], [], [[x0,y0]]
                have_visited[x0, y0] = 1
                
                while(len(queue) > 0):
                    x, y = queue.pop()
                    gr_list.append([x,y])
                    for dx, dy in search_dir:
                        if( check_inside_bound(x+dx, y+dy, f_dim) == False ):
                            continue
                        
                        if( have_visited[x+dx, y+dy] == 0 and fg_images[frame_num][x+dx, y+dy] > 0 ):
                            queue.append([x+dx, y+dy])
                            have_visited[x+dx, y+dy] = 1
    
                groups.append({'frame':frame_num, 'elems':gr_list})
                
    return groups

def keep_only_in_group(mat, group_elems, f_dim):
    th = 0.1
    ind_list = []
    filtered_mat = np.zeros_like(mat)
    for x,y in group_elems:
        filtered_mat[x,y] = mat[x,y]
        if( mat[x,y] > th ):
            ind_list.append(utils.index2d_to_1d(x,y, f_dim))
            
    return filtered_mat, ind_list
