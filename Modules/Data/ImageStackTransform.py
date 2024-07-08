"""
FILENAME: ImageStackTransform.py
DESCRIPTION: Define transforms for a stack of images 
@author: Jian Zhong
"""


import torch
import torchvision.transforms.functional as F
from torchvision.transforms import v2


# random crop function for image stack
class RandomCrop(v2.RandomCrop):
    def __init__(
            self,
            **args
    ):
        """
        Refer v2.RandomCrop documentation for agrument definition
        """
        super().__init__(**args)

    def forward(self, src_image_stack):
        dst_image_stack = [None for _ in range(len(src_image_stack))]
        
        i, j, h, w = 0, 0, 0, 0

        for i_img in range(len(src_image_stack)):
            cur_src_image = src_image_stack[i_img]

            if self.padding is not None:
                cur_src_image = F.pad(cur_src_image, self.padding, self.fill, self.padding_mode)

            _, height, width = F.get_dimensions(cur_src_image)
            # pad the width if needed
            if self.pad_if_needed and width < self.size[1]:
                padding = [self.size[1] - width, 0]
                cur_src_image = F.pad(cur_src_image, padding, self.fill, self.padding_mode)
            # pad the height if needed
            if self.pad_if_needed and height < self.size[0]:
                padding = [0, self.size[0] - height]
                cur_src_image = F.pad(cur_src_image, padding, self.fill, self.padding_mode)

            if i_img == 0:
                i, j, h, w = self.get_params(cur_src_image, self.size)

            dst_image_stack[i_img] = F.crop(cur_src_image, i, j, h, w)
        
        return dst_image_stack
    

# random horizontal flip
class RandomHorizontalFlip(torch.nn.Module):

    def __init__(self, p = 0.5):
        """
        p (float): probability of the image being flipped.
        """
        super().__init__()
        self.p = p
    
    def forward(self,src_image_stack):
        dst_image_stack = src_image_stack
        if torch.rand(1) < self.p:
            dst_image_stack = [None for _ in range(len(src_image_stack))]
            for i_img in range(len(src_image_stack)):
                cur_src_image = src_image_stack[i_img]
                dst_image_stack[i_img] = F.hflip(cur_src_image)

        return dst_image_stack
    

# random vertial flip
class RandomVerticalFlip(torch.nn.Module):

    def __init__(self, p = 0.5):
        """
        p (float): probability of the image being flipped.
        """
        super().__init__()
        self.p = p
    
    def forward(self,src_image_stack):
        dst_image_stack = src_image_stack
        if torch.rand(1) < self.p:
            dst_image_stack = [None for _ in range(len(src_image_stack))]
            for i_img in range(len(src_image_stack)):
                cur_src_image = src_image_stack[i_img]
                dst_image_stack[i_img] = F.vflip(cur_src_image)

        return dst_image_stack
    

# Elastic transform
class ElasticTransform(v2.ElasticTransform):
    def __init__(
            self,
            fills = [0, 0],
            **args,
    ):
        """
        Refer v2.ElasticTransform documentation for agrument definition
        """
        super().__init__(**args)
        for i_fill in range(len(fills)):
            if isinstance(fills[i_fill], (int,float)):
                fills[i_fill] = [float(fills[i_fill])]
            elif isinstance(fills[i_fill], (list, tuple)):
                fills[i_fill] = [float(f) for f in fills[i_fill]]
            elif isinstance(fills[i_fill], str):
                continue
            else:
                raise TypeError(f"fill should be int or float or a list or tuple of them. Got {type(fill)}")
        self.fills = fills 

    def forward(self, src_image_stack):
        _, height, width = F.get_dimensions(src_image_stack[0])
        displacement = self.get_params(self.alpha, self.sigma, [height, width])
        
        dst_image_stack = [None for _ in range(len(src_image_stack))]
        for i_img in range(len(src_image_stack)):
            cur_src_image = src_image_stack[i_img]
            
            reshape_ichannel = False
            if len(cur_src_image.size()) == 2:
                cur_src_image = torch.unsqueeze(cur_src_image, dim = 0)
                reshape_ichannel = True

            cur_fill = self.fills[i_img]
            channels, _, _ = F.get_dimensions(cur_src_image)
            if isinstance(cur_src_image, torch.Tensor):
                if isinstance(cur_fill, str):
                    if cur_fill == "mean":
                        cur_fill = [0 for _ in range(channels)]
                        for i_chan in range(channels):
                            cur_fill[i_chan] = torch.mean(cur_src_image[i_chan,...])
                    elif cur_fill == "median":
                        cur_fill = [0 for _ in range(channels)]
                        for i_chan in range(channels):
                            cur_fill[i_chan] = torch.median(cur_src_image[i_chan,...])
                    elif cur_fill == "min":
                        cur_fill = [0 for _ in range(channels)]
                        for i_chan in range(channels):
                            cur_fill[i_chan] = torch.min(cur_src_image[i_chan,...])                    
                    elif cur_fill == "max":
                        cur_fill = [0 for _ in range(channels)]
                        for i_chan in range(channels):
                            cur_fill[i_chan] = torch.max(cur_src_image[i_chan,...])    
                        
            dst_image = F.elastic_transform(cur_src_image, displacement, self.interpolation, cur_fill)
            if reshape_ichannel:
                dst_image = torch.squeeze(dst_image, dim = 0)
            dst_image_stack[i_img] = dst_image

        return dst_image_stack
    

## Random rotations
class RandomRotation(v2.RandomRotation):
    def __init__(
            self,
            fills = [0, 0],
            **args, 
    ):
        """
        Refer v2.RandomRotation documentation for agrument definition
        """
        super().__init__(**args)
        self.fills = fills


    def forward(self, src_image_stack):
        angle = self.get_params(self.degrees)

        dst_image_stack = [None for _ in range(len(src_image_stack))]
        for i_img in range(len(src_image_stack)):
            cur_src_image = src_image_stack[i_img]

            reshape_ichannel = False
            if len(cur_src_image.size()) == 2:
                cur_src_image = torch.unsqueeze(cur_src_image, dim = 0)
                reshape_ichannel = True

            cur_fill = self.fills[i_img]
            channels, _, _ = F.get_dimensions(cur_src_image)
            if isinstance(cur_src_image, torch.Tensor):
                if isinstance(cur_fill, str):
                    if cur_fill == "mean":
                        cur_fill = [0 for _ in range(channels)]
                        for i_chan in range(channels):
                            cur_fill[i_chan] = torch.mean(cur_src_image[i_chan,...])
                    elif cur_fill == "median":
                        cur_fill = [0 for _ in range(channels)]
                        for i_chan in range(channels):
                            cur_fill[i_chan] = torch.median(cur_src_image[i_chan,...])
                    elif cur_fill == "min":
                        cur_fill = [0 for _ in range(channels)]
                        for i_chan in range(channels):
                            cur_fill[i_chan] = torch.min(cur_src_image[i_chan,...])                    
                    elif cur_fill == "max":
                        cur_fill = [0 for _ in range(channels)]
                        for i_chan in range(channels):
                            cur_fill[i_chan] = torch.max(cur_src_image[i_chan,...])     
                else:
                    if isinstance(cur_fill, (int,float)):
                        cur_fill = [float(cur_fill)] * channels
                    else:
                        cur_fill = [float(f) for f in cur_fill]

            dst_image = F.rotate(cur_src_image, angle, self.interpolation, self.expand, self.center, cur_fill)
            if reshape_ichannel:
                dst_image = torch.squeeze(dst_image, dim = 0)
                
            dst_image_stack[i_img] = dst_image

        return dst_image_stack

