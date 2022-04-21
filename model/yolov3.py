import torch
import torch.nn as nn
import os

"""
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import json
import cv2
import torchvision

from torchvision import transforms

from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import math

from ..utils import inverse_sigmoid, intersection, get_cell_offsets
from ..utils import YOLOV3
from ..utils import label_map, priors, scales
"""


class YoloConv(nn.Module):
    def __init__(self, in_size, out_size):

        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(out_size, in_size, kernel_size=(1, 1))
        self.conv2 = nn.Conv2d(
            in_size, out_size, kernel_size=(3, 3), padding=(1, 1))
        self.bn = nn.BatchNorm2d(out_size)
        self.relu = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x_f = self.conv1(x)
        x_f = self.conv2(x_f)
        x_f = self.bn(x_f)
        x_f = self.relu(x_f)

        x = x + x_f

        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_size, out_size):

        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(out_size, in_size, kernel_size=(1, 1))
        self.conv2 = nn.Conv2d(
            in_size, out_size, kernel_size=(3, 3), padding=(1, 1))
        self.bn = nn.BatchNorm2d(out_size)
        self.relu = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x_f = self.conv1(x)
        x_f = self.conv2(x_f)
        x_f = self.bn(x_f)
        x_f = self.relu(x_f)

        x = x + x_f

        return x


class YoloHead1(nn.Module):
    def __init__(self, sz_1, sz_2):
        super(YoloHead1, self).__init__()
        # sz_2 should be smaller
        self.conv1 = torch.nn.Conv2d(sz_1, sz_2, 1)
        self.conv2 = torch.nn.Conv2d(sz_2, sz_1, 3, padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(sz_2)
        self.bn2 = nn.BatchNorm2d(sz_1)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.bn2(x)
        return x


def generate_conv(layer: dict, in_channels):

    filters = layer['filters']
    stride = layer['stride']
    pad = layer['pad']
    kernel_size = layer['size']
    activation = layer['activation']

    if kernel_size == 1:
        pad = 0

    conv = nn.Conv2d(in_channels, filters,
                     kernel_size=kernel_size, stride=stride, padding=pad)

    return conv


class YoloHead(nn.Module):
    def __init__(self):
        super(YoloHead, self).__init__()

    def forward(self, x):
        return x


class YOLOV3(nn.Module):
    def __init__(self, cfg):
        # input size = (256, 256)
        super(YOLOV3, self).__init__()

        params = cfg['params']
        cfg_layers = cfg['layers']
        layers = []
        im_channels = 3
        prev_conv_inc = None

        self.shortcuts = dict()
        saved_x = dict()

        self.routes = dict()

        i = 0
        for layer in cfg['layers']:
            if layer['name'] == 'convolutional':
                if prev_conv_inc == None:
                    conv_layer = generate_conv(layer, im_channels)
                else:
                    conv_layer = generate_conv(layer, prev_conv_inc)

                curr_layer = []
                curr_layer.append(conv_layer)

                prev_conv_inc = layer['filters']

                if 'batch_normalize' in layer.keys() and layer['batch_normalize'] == 1:
                    bn_layer = nn.BatchNorm2d(layer['filters'])
                    curr_layer.append(bn_layer)

                if layer['activation'] == 'leaky':
                    relu_layer = nn.LeakyReLU()
                    curr_layer.append(relu_layer)
                elif layer['activation'] == 'relu':
                    relu_layer = nn.ReLU()
                    curr_layer.append(relu_layer)

                curr_layer = nn.Sequential(*curr_layer)

                layers.append(curr_layer)
                i += 1

            elif layer['name'] == 'upsample':
                upsample_layer = torch.nn.Upsample(scale_factor=2)
                layers.append(upsample_layer)
                i += 1

            elif layer['name'] == 'shortcut':
                _from = int(layer['from'])
                if _from < 1:
                    _from = i - _from
                _to = i

                self.shortcuts[_from] = _to
                i += 1

            elif layer['name'] == 'route':
                if ',' in layer['layers']:
                    route_layers = layer['layers'].split(',')
                    _to = route_layers[0]
                    _from = route_layers[1]
                    self.routes[_from] = _to
                else:
                    _from = int(layer['layers'])
                    if _from < 1:
                        _from = i - _from
                    _to = i

                    self.routes[_from] = _to
                i += 1

            elif layer['name'] == 'yolo':
                layers.append(YoloHead())
                i += 1

        self.layers = nn.Sequential(*layers)

    def summary(self):

        for i, layer in enumerate(self.layers):
            print(f'{i}: {layer}')

            if i == 10:
                break

    def load_weights(self, weights_file):
        return NotImplemented

    def forward(self, x):

        saved_x_shortcuts = dict()
        saved_x_routes = dict()

        for i, layer in enumerate(self.layers):

            x = layer(x)
            if i in self.shortcuts.keys():
                _to = self.shortcuts[i]
                saved_x_shortcuts[_to] = x
            elif i in saved_x_shortcuts.keys():
                x = x + saved_x_shortcuts[i]

            if i in self.routes.keys():
                _to = self.routes[i]
                saved_x_routes[_to] = x
            elif i in saved_x_routes.keys():
                x = torch.cat(x, saved_x_routes[i])\

        return x
