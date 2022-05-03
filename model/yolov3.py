import torch
import torch.nn as nn
import os

import numpy as np
import os
"""
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



def process_outputs(outputs: list):
    """given raw yolo outputs, generate a list of bounding boxes

    Args:
        outputs (list): _description_
    """

    iou = 0.5
    threshold = 0.5







def generate_conv(layer: dict, in_channels, bias=False):

    filters = layer['filters']
    stride = layer['stride']
    pad = layer['pad']
    kernel_size = layer['size']
    activation = layer['activation']

    if kernel_size == 1:
        pad = 0

    conv = nn.Conv2d(in_channels, filters,
                     kernel_size=kernel_size, stride=stride, padding=pad, bias=bias)

    return conv


class YoloHead(nn.Module):
    def __init__(self):

        super(YoloHead, self).__init__()

    def forward(self, x):
        # do fancy math here
        return x



class YoloRoute(nn.Module):
    def __init__(self):

        super(YoloRoute, self).__init__()

    def forward(self, x):
        # do fancy math here
        return x


class YoloShortcut(nn.Module):
    def __init__(self):

        super(YoloShortcut, self).__init__()

    def forward(self, x):
        # do fancy math here
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
        self.single_routes = dict()

        self.yolo_layers = []

        i = 1
        l = 0


        for layer in cfg['layers']:
            if layer['name'] == 'convolutional':

                bias = True

                if 'batch_normalize' in layer.keys():
                    bias = False 

                if prev_conv_inc == None:
                    conv_layer = generate_conv(layer, im_channels, bias=bias)
                else:
                    conv_layer = generate_conv(layer, prev_conv_inc, bias=bias)

                curr_layer = []
                curr_layer.append(conv_layer)

                prev_conv_inc = layer['filters']

                if 'batch_normalize' in layer.keys() and layer['batch_normalize'] == 1:
                    bn_layer = nn.BatchNorm2d(layer['filters'])
                    curr_layer.append(bn_layer)

                if layer['activation'] == 'leaky':
                    relu_layer = nn.LeakyReLU()
                    curr_layer.append(relu_layer)
                    no_bias = False
                elif layer['activation'] == 'relu':
                    relu_layer = nn.ReLU()
                    curr_layer.append(relu_layer)

                curr_layer = nn.Sequential(*curr_layer)

                layers.append(curr_layer)
                i += 1
                l += 1

            elif layer['name'] == 'upsample':
                upsample_layer = torch.nn.Upsample(scale_factor=2)
                layers.append(upsample_layer)
                i += 1
                l += 1

            elif layer['name'] == 'shortcut':
                _from = int(layer['from'])
                if _from < 1:
                    _from = i - _from
                _to = i

                self.shortcuts[_from] = _to
                layers.append(YoloShortcut())
                i += 1

            elif layer['name'] == 'route':

                # the route has 2 layers, we concatenate
                if isinstance(layer['layers'], list):
                    _from = int(layer['layers'][0])
                    if _from < 1:
                        _from = i + _from
                    _to = int(layer['layers'][1])

                    self.routes[_to] = _from

                else:
                    _from = int(layer['layers'])
                    if _from < 1:
                        _from = i + _from 
                    _to = i

                    self.single_routes[_from] = _to

                layers.append(YoloRoute())
                i += 1

            elif layer['name'] == 'yolo':

                layers.append(YoloHead())
                self.yolo_layers.append(i)
                i += 1
                l += 1
            

        for i in self.routes.keys():
            _from = self.routes[i]
            conv_shape_in = layers[i - 1][0].weight.shape[1] + layers[_from - 2][0].weight.shape[0] * 2
            conv_shape_out = layers[_from - 2][0].weight.shape[0]
            layers[_from + 1][0] = nn.Conv2d(conv_shape_in, conv_shape_out, 1, 1, bias=False) 

        for i in self.single_routes.keys():

            _from = self.single_routes[i]
            conv_shape_in = layers[i][0].weight.shape[1]
            conv_shape_out = layers[_from][0].weight.shape[0]
            layers[_from][0] = nn.Conv2d(conv_shape_in, conv_shape_out, 1, 1, bias=False) 


        self.layers = nn.Sequential(*layers)

    def summary(self):
        for i, layer in enumerate(self.layers):
            print(f'{i}: {layer}')

    def getexp(self, param):
        return param.split('.')[:3]

    def load_weights(self, weights_file):

        ptr = 0
        i = 0
        even = 0
        prev = []

        sd = self.state_dict()
        sd_keys = list(sd.keys())


        len_sd = len(sd_keys)

        with open(weights_file, 'rb') as f:
            header = np.fromfile(f, dtype=np.int32, count=5)
            weights = np.fromfile(f, dtype=np.float32)

        ptr = 0


        for layer in self.layers:

            layer_type = type(layer)

            if isinstance(layer, torch.nn.modules.container.Sequential):

                # print(layer)
                # handle the conv layer first
                conv = layer[0]

                if conv.bias is not None:
                    bs = conv.bias.data.shape.numel()
                    conv.bias.data = torch.from_numpy(
                        weights[ptr:ptr+bs])
                    ptr += bs
                
                ws = tuple(conv.weight.data.shape)
                wsz = np.prod(ws)
               # beta (weight)
                conv.weight.data = torch.from_numpy(
                    weights[ptr:ptr+wsz]).view_as(conv.weight.data)
                ptr += wsz

                if len(layer) > 1:

                    bn = layer[1]
                    bs = bn.bias.data.shape.numel()

                    bn.bias.data = torch.from_numpy(weights[ptr:ptr+bs])
                    ptr += bs
                    # beta (weight)
                    bn.weight.data = torch.from_numpy(weights[ptr:ptr+bs])
                    ptr += bs
                    # running mean
                    bn.running_mean.data = torch.from_numpy(
                        weights[ptr:ptr+bs])
                    ptr += bs
                    # running variance
                    bn.running_var.data = torch.from_numpy(weights[ptr:ptr+bs])
                    ptr += bs
                    # sd[param_num_batches_tracked] =  torch.from_numpy(weights[ptr:ptr+bs])
                    # ptr += bs

        print(f'weights loaded: {ptr}')

    def forward(self, x):

        saved_x_shortcuts = dict()
        saved_x_routes = dict()
        saved_single_x_routes = dict()
        yolo_outputs = []

        for i, layer in enumerate(self.layers):

            if i in self.yolo_layers: 
                x = layer(x)
                yolo_outputs.append(x)
            else:
                if i in self.single_routes.keys():
                    _to = self.single_routes[i]
                    saved_single_x_routes[_to] = x
                elif i in saved_single_x_routes.keys():
                    x = saved_single_x_routes[i]
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
                    x = torch.cat((x, saved_x_routes[i]), 1)
        
        yolo_outputs.append(x)


        for i in range(len(yolo_outputs)):
            yolo_outputs[i][:, :2, :, :] = 


        if self.process_outputs:
            return process_outputs(yolo_outputs)
        else:
            return yolo_outputs
