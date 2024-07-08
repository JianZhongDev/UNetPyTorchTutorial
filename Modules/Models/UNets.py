"""
FILENAME: Unet.py
DESCRIPTION: Unet model definition
@author: Jian Zhong
"""

import torch
from torch import nn
from .Layers import (DebugLayers, StackedLayers, ResidualLayers)


class Simple3LayerUNet(nn.Module):
    def __init__(
            self,
            in_channels = 3,
            out_channels = 1,
            layer_nof_channels = [32, 64, 128, 256, 512],
            activation = None,
            padding_mode = "reflect", # zero padding tends to have zeros boundary, which greatly affects learning 
    ):
        super().__init__()

        assert len(layer_nof_channels) == 5

        if activation is None:
            activation = nn.LeakyReLU

        self.network = nn.Sequential(
            StackedLayers.Stacked2DConv(
                [{"nof_layers": 3, "in_channels": in_channels, "out_channels": layer_nof_channels[0], "activation": activation, "padding_mode": padding_mode,}],
            ),
            ResidualLayers.ResdiualCat(
                nn.Sequential(
                    nn.MaxPool2d(kernel_size = 2, stride = 2),
                    StackedLayers.Stacked2DConv(
                        [{"nof_layers": 3, "in_channels": layer_nof_channels[0], "out_channels": layer_nof_channels[1], "activation": activation, "padding_mode": padding_mode,}],
                    ),
                    ResidualLayers.ResdiualCat(
                        nn.Sequential(
                            nn.MaxPool2d(kernel_size = 2, stride = 2),
                            StackedLayers.Stacked2DConv(
                                [{"nof_layers": 3, "in_channels": layer_nof_channels[1], "out_channels": layer_nof_channels[2], "activation": activation, "padding_mode": padding_mode,}],
                            ),
                            ResidualLayers.ResdiualCat(
                                nn.Sequential(
                                    nn.MaxPool2d(kernel_size = 2, stride = 2),
                                    StackedLayers.Stacked2DConv(
                                        [
                                            {"nof_layers": 3, "in_channels": layer_nof_channels[2], "out_channels": layer_nof_channels[3], "activation": activation, "padding_mode": padding_mode,},
                                            {"nof_layers": 3, "out_channels": layer_nof_channels[4], "activation": activation, "padding_mode": padding_mode,},
                                            {"nof_layers": 3, "out_channels": layer_nof_channels[3], "activation": activation, "padding_mode": padding_mode,},
                                         ],
                                    ),
                                    nn.ConvTranspose2d(
                                        in_channels = layer_nof_channels[3],
                                        out_channels = layer_nof_channels[2],
                                        kernel_size = 2, stride = 2, 
                                    ),
                                ),
                            ),
                            StackedLayers.Stacked2DConv(
                                [{"nof_layers": 3, "in_channels": 2 * layer_nof_channels[2], "out_channels": layer_nof_channels[2], "activation": activation, "padding_mode": padding_mode,}],
                            ),
                            nn.ConvTranspose2d(
                                in_channels = layer_nof_channels[2],
                                out_channels = layer_nof_channels[1],
                                kernel_size = 2, stride = 2, 
                            ),
                        )
                    ),
                    StackedLayers.Stacked2DConv(
                        [{"nof_layers": 3, "in_channels": 2 * layer_nof_channels[1], "out_channels": layer_nof_channels[1], "activation": activation, "padding_mode": padding_mode,}],
                    ),
                    nn.ConvTranspose2d(
                        in_channels = layer_nof_channels[1],
                        out_channels = layer_nof_channels[0],
                        kernel_size = 2, stride = 2, 
                    ),
                ),    
            ),
            StackedLayers.Stacked2DConv(
                [
                    {"nof_layers": 3, "in_channels": 2 * layer_nof_channels[0], "out_channels": layer_nof_channels[0], "activation": activation, "padding_mode": padding_mode,},
                    {"nof_layers": 1, "out_channels": out_channels, "kernel_size": 1, "padding": 0, "activation": None, "padding_mode": padding_mode,},
                ],
            ),
            # nn.Softmax(dim = 1),
        )

    def forward(self, x):
        return self.network(x)