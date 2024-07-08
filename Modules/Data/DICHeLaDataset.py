"""
FILENAME: DICHeLaSegDataset.py
DESCRIPTION: PyTorch dataset for DIC HeLa Dataset
@author: Jian Zhong
"""

import os
import glob
import torch
import numpy as np
import cv2
from torch import nn
from torch.utils.data import Dataset
from torchvision.io import read_image


## DIC HeLa Dataset
## Source Dataset webiste: https://celltrackingchallenge.net/2d-datasets/
class DICHeLaSegDataset(Dataset):

    ## helper function to read image
    def __read_image(self, src_image_path):
        cur_image = None
        if os.path.splitext(src_image_path) == ".png" or  os.path.splitext(src_image_path) == ".jpg":
            cur_image = read_image(src_image_path)
        else:
            cur_image = cv2.imread(src_image_path, -1)
            if(cur_image.dtype == np.uint16):
                cur_image = cur_image.astype(np.int32)
            cur_image = torch.from_numpy(cur_image)
            if(len(cur_image.size()) < 3):
                cur_image = cur_image.view((1,) + tuple(cur_image.size()))
        return cur_image

    def __init__(
            self,
            data_image_path_globs = [],
            seg_image_path_globs = None,
            data_transform = None,
            target_transform = None,
            common_transform = None,
            color_categories = False,
        ):
        # naming of the data image and the corresponding segmentation image should make their index in glob() the same 

        self.data_image_paths = []
        for cur_path_glob in data_image_path_globs:
            self.data_image_paths += glob.glob(cur_path_glob)

        self.seg_image_paths = None
        if seg_image_path_globs is not None:
            self.seg_image_paths = []
            for cur_path_glob in seg_image_path_globs:
                self.seg_image_paths += glob.glob(cur_path_glob)

            assert len(self.data_image_paths) == len(self.seg_image_paths)
        
        self.data_transform = data_transform
        self.target_transform = target_transform
        self.common_transform = common_transform 
        self.color_categories = color_categories


    def __len__(self):
        return len(self.data_image_paths)
        
    def __getitem__(self, idx):

        # get image and target tensors
        cur_data = None
        cur_data_image_path = self.data_image_paths[idx]
        cur_data = self.__read_image(cur_data_image_path)

        cur_target = None
        if self.seg_image_paths is not None:
            cur_seg_image_path = self.seg_image_paths[idx]
            cur_target = self.__read_image(cur_seg_image_path)
            if not self.color_categories:
                cur_target[cur_target > 0] = 1
        else:
            cur_target = torch.full_like(cur_data, 0)

        # apply transforms
        if self.data_transform is not None:
            cur_data = self.data_transform(cur_data)
        if self.target_transform is not None:
            cur_target = self.target_transform(cur_target)
        if self.common_transform is not None:
            cur_data_pkg = [cur_data, cur_target]
            cur_data_pkg = self.common_transform(cur_data_pkg)
            cur_data = cur_data_pkg[0]
            cur_target = cur_data_pkg[1]

        return cur_data, cur_target


## DIC HeLa Dataset with preprocessed weights
## Source Dataset webiste: https://celltrackingchallenge.net/2d-datasets/
class DICHeLaWeightedSegDataset(Dataset):

    ## helper function to read image
    def __read_image(self, src_image_path):
        cur_image = None
        if os.path.splitext(src_image_path) == ".png" or  os.path.splitext(src_image_path) == ".jpg":
            cur_image = read_image(src_image_path)
        else:
            cur_image = cv2.imread(src_image_path, -1)
            if(cur_image.dtype == np.uint16):
                cur_image = cur_image.astype(np.int32)
            cur_image = torch.from_numpy(cur_image)
            if(len(cur_image.size()) < 3):
                cur_image = cur_image.view((1,) + tuple(cur_image.size()))
        return cur_image


    def __init__(
            self,
            data_image_path_globs = [],
            seg_image_path_globs = None,
            seg_weight_path_globs = None,
            data_transform = None,
            target_transform = None,
            weight_transform = None,
            common_transform = None,
            color_categories = False,
        ):
        # naming of the data image and the corresponding segmentation image should make their index in glob() the same 

        self.data_image_paths = []
        for cur_path_glob in data_image_path_globs:
            self.data_image_paths += glob.glob(cur_path_glob)

        self.seg_image_paths = None
        if seg_image_path_globs is not None:
            self.seg_image_paths = []
            for cur_path_glob in seg_image_path_globs:
                self.seg_image_paths += glob.glob(cur_path_glob)

            assert len(self.data_image_paths) == len(self.seg_image_paths)

        self.seg_weight_paths = None
        if seg_weight_path_globs is not None:
            self.seg_weight_paths = []
            for cur_path_glob in seg_weight_path_globs:
                self.seg_weight_paths += glob.glob(cur_path_glob)
        
        self.data_transform = data_transform
        self.target_transform = target_transform
        self.weight_transform = weight_transform
        self.common_transform = common_transform 
        self.color_categories = color_categories


    def __len__(self):
        return len(self.data_image_paths)
        
    def __getitem__(self, idx):

        # get image and target tensors
        cur_data = None
        cur_data_image_path = self.data_image_paths[idx]
        cur_data = self.__read_image(cur_data_image_path)

        cur_target = None
        if self.seg_image_paths is not None:
            cur_seg_image_path = self.seg_image_paths[idx]
            cur_target = self.__read_image(cur_seg_image_path)
            if not self.color_categories:
                cur_target[cur_target > 0] = 1
        else:
            cur_target = torch.full_like(cur_data, 0)

        cur_weight = None
        if self.seg_weight_paths is not None:
            cur_seg_weight_path = self.seg_weight_paths[idx]
            cur_weight = self.__read_image(cur_seg_weight_path)
        else:
            cur_weight = torch.full_like(cur_data, 0)

        # apply transforms
        if self.data_transform is not None:
            cur_data = self.data_transform(cur_data)
        if self.target_transform is not None:
            cur_target = self.target_transform(cur_target)
        if self.weight_transform is not None:
            cur_weight = self.weight_transform(cur_weight)
        if self.common_transform is not None:
            cur_data_pkg = [cur_data, cur_target, cur_weight]
            cur_data_pkg = self.common_transform(cur_data_pkg)
            cur_data = cur_data_pkg[0]
            cur_target = cur_data_pkg[1]
            cur_weight = cur_data_pkg[2]

        return cur_data, cur_target, cur_weight



