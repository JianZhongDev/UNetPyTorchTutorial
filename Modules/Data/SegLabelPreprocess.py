"""
FILENAME: SegLabelPreprocess.py
DESCRIPTION: Preprocessing script for segmentation labels
@author: Jian Zhong
"""

import numpy as np
import cv2


## erode label masks
def erode_colored_labels(src_colored_labels, kernel = None, nof_itrs = 1, bkg_color = 0):
    if kernel is None:
        kernel = np.ones((3,3), dtype = src_colored_labels.dtype)

    label_colors = np.unique(src_colored_labels[src_colored_labels != bkg_color])
    dst_colored_labels = np.full(src_colored_labels.shape, bkg_color, dtype = src_colored_labels.dtype) 
    for cur_color in label_colors:
        cur_mask = np.zeros_like(src_colored_labels)
        cur_mask[src_colored_labels == cur_color] = 1
        cur_mask = cv2.erode(cur_mask, kernel = kernel, iterations = nof_itrs)
        dst_colored_labels[cur_mask > 0] = cur_color

    return dst_colored_labels


## class weight balance weight
def balanced_weight_colored_labels(src_colored_labels, mode = "median"):
    
    colors = np.unique(src_colored_labels)
    nof_pxls = src_colored_labels.size

    balanced_weights = np.zeros(src_colored_labels.shape, dtype = float)
    for cur_color in colors:
        cur_mask = src_colored_labels == cur_color
        balanced_weights[cur_mask] = np.sum(cur_mask)

    if mode == "median":
        balanced_weights = np.median(np.unique(balanced_weights))/balanced_weights
    else:
        balanced_weights = nof_pxls/balanced_weights
    
    return balanced_weights


## calculate the distance of each background pixel to label border 
def border_distance_colored_labels(src_colored_labels, bkg_color = 0):

    label_colors = np.unique(src_colored_labels[src_colored_labels != bkg_color])
    
    center_erode_kernel = np.ones((3,3), dtype = src_colored_labels.dtype)
    center_erode_nof_itrs = 1

    src_label_centers = erode_colored_labels(
        src_colored_labels, 
        kernel = center_erode_kernel,
        nof_itrs = center_erode_nof_itrs,
        bkg_color = bkg_color,
        )
    src_label_border = src_colored_labels.copy()
    src_label_border[src_label_centers != bkg_color] = bkg_color

    nof_colors = len(label_colors)
    image_height = src_colored_labels.shape[-2]
    image_width = src_colored_labels.shape[-1]

    label_border_distances = np.zeros((nof_colors, image_height, image_width), dtype = float)
    for i_color in range(nof_colors):
        cur_color = label_colors[i_color]
        cur_border_is, cur_border_js = np.where(src_label_border == cur_color)
        for i in range(image_height):
            for j in range(image_width):
                if src_colored_labels[i,j] != bkg_color:
                    continue
                cur_border_distance = np.min(np.sqrt( (i - cur_border_is)**2 + (j - cur_border_js)**2))
                label_border_distances[i_color, i, j] = cur_border_distance

    return label_border_distances, label_colors


## calculate broder weights
def border_distance_gaussian_weight(src_colored_labels, bkg_color = 0, sigma = 5, topk = 2):
    
    label_border_distances, label_colors  = border_distance_colored_labels(src_colored_labels, bkg_color)
    topk = min(topk, label_border_distances.shape[0])
    label_border_distances_topk_sum = np.sum(np.sort(label_border_distances, axis = 0)[:topk,:,:], axis = 0)

    border_distance_gaussian_weight = np.exp(- (label_border_distances_topk_sum**2)/(2 * sigma ** 2) )
    border_distance_gaussian_weight[src_colored_labels != bkg_color] = 0

    return border_distance_gaussian_weight
    



