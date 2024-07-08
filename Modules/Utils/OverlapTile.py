"""
FILENAME: OverlapTile.py
DESCRIPTION: Implemenationn of overlap tile stragety 
@author: Jian Zhong
"""

import torch
import math


## split source image into sub images
def split(src_image, sub_image_size, stride):
    
    nof_channels, src_image_height, src_image_width = src_image.size()
    sub_image_height, sub_image_width = sub_image_size
    if isinstance(stride, int):
        stride = [stride for _ in range(2)]
    row_stride, col_stride = stride

    assert src_image_height >= sub_image_height
    assert src_image_width >= sub_image_width
    assert src_image_height >= row_stride
    assert src_image_width >=col_stride

    nof_rows = math.ceil(src_image_height/row_stride)
    nof_cols = math.ceil(src_image_width/col_stride)

    dst_sub_images = torch.zeros((nof_rows*nof_cols, nof_channels, sub_image_height, sub_image_width),dtype = src_image.dtype)
    dst_sub_image_locs = torch.zeros((nof_rows*nof_cols, 2), dtype = int)

    # iterate through each row and column and split the image into sub images 
    for i_row in range(nof_rows):
        cur_row_start = i_row * row_stride
        cur_row_start = min(cur_row_start, src_image_height - sub_image_height)
        cur_row_end = cur_row_start + sub_image_height
        for i_col in range(nof_cols):
            cur_col_start = i_col * col_stride
            cur_col_start = min(cur_col_start, src_image_width - sub_image_width)
            cur_col_end = cur_col_start + sub_image_width
            cur_idx = i_row * nof_cols + i_col
            dst_sub_images[cur_idx,...] = src_image[...,cur_row_start:cur_row_end,cur_col_start:cur_col_end]
            dst_sub_image_locs[cur_idx, 0] = cur_row_start
            dst_sub_image_locs[cur_idx, 1] = cur_col_start
    
    return dst_sub_images, dst_sub_image_locs


## merge sub images into source images
def merge(src_sub_images, sub_image_locs, dst_image_size, reduction = "mode"):

    nof_sub_images, nof_channels, sub_image_height, sub_image_width = src_sub_images.size()
    src_image_height, src_image_width = dst_image_size

    # calculate indexs of each sub image pixels in dst image
    src_image_is, src_image_js = torch.meshgrid(
        torch.arange(src_image_height), 
        torch.arange(src_image_width),
        indexing = "ij",
    )

    sub_image_is = torch.zeros((nof_sub_images, sub_image_height, sub_image_width), dtype = int)
    sub_image_js = torch.zeros((nof_sub_images, sub_image_height, sub_image_width), dtype = int)

    for i_subimg in range(nof_sub_images):
        cur_row_start, cur_col_start = sub_image_locs[i_subimg,:]
        cur_row_end = cur_row_start + sub_image_height
        cur_col_end = cur_col_start + sub_image_width
        
        sub_image_is[i_subimg,...] = src_image_is[cur_row_start:cur_row_end, cur_col_start:cur_col_end]
        sub_image_js[i_subimg,...] = src_image_js[cur_row_start:cur_row_end, cur_col_start:cur_col_end]

    dst_image = torch.zeros((nof_channels, src_image_height, src_image_width), dtype = src_sub_images.dtype)

    # iterate through each pixel and process overlapped pixels if necessary
    src_sub_images = torch.swapaxes(src_sub_images, 0, 1)
    for i in range(src_image_height):
        for j in range(src_image_width):
            cur_mask = torch.logical_and(sub_image_is == i, sub_image_js == j)
            cur_cands = src_sub_images[..., cur_mask]
            if reduction == "mode":
                mode_vals, mode_idxs = torch.mode(cur_cands, dim = -1)
                dst_image[:,i,j] = mode_vals
            elif reduction == "min":
                dst_image[:,i,j] = torch.min(cur_cands, dim = -1)
            elif reduction == "max":
                dst_image[:,i,j] = torch.max(cur_cands, dim = -1)
    src_sub_images = torch.swapaxes(src_sub_images, 0, 1)
    
    return dst_image
