from pyrsistent import b
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import json
import cv2
import torchvision
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision import transforms
import math


# call_x, cell_y are the distance of the cell from the top right corner. Multiply by 32
def get_gt(label: torch.Tensor, cell_x, cell_y, prior: tuple, scale=32) -> torch.Tensor:
    gt = torch.zeros(5)

    gt[1] = label[1]
    gt[2] = label[2]
    gt[3] = label[3]
    gt[4] = label[4]

    prior_w, prior_h = prior
    prior_x = cell_x + (scale // 2)
    prior_y = cell_y + (scale // 2)

    gt[1] = inverse_sigmoid((gt[1] - cell_x) / 32)
    gt[2] = inverse_sigmoid((gt[2] - cell_y) / 32)
    gt[2] /= 32

    gt[3] = gt[3] / prior_w
    gt[3] /= 32
    gt[4] = gt[4] / prior_h
    gt[4] /= 32

    gt[3] = torch.log(gt[3])
    gt[4] = torch.log(gt[4])

    # inverse sigmoid of the ioU score
    _iou = iou(prior_w, prior_h, label[3], label[4])
    if _iou == 0:
        o = 0
    else:
        o = torch.log(_iou)
    return torch.tensor((*gt, o))


def coco2yolo(label: torch.Tensor) -> torch.Tensor:
    yolo = torch.clone(label)

    yolo[1] = yolo[1] + (yolo[3] / 2)
    yolo[2] = yolo[2] + (yolo[4] / 2)
    return yolo


def xywh2xyxy(arr: torch.Tensor) -> torch.Tensor:
    x0 = arr[0] - (arr[2] / 2)
    y0 = arr[1] - (arr[3] / 2)
    x1 = x0 + arr[2]
    y1 = y0 + arr[3]

    return torch.Tensor((x0, y0, x1, y1))


def compare_iou(a, b):

    # a, b: xyxy
    # this should not be negative

    # this is stupid, fix later

    i_w = abs(min(a[2], b[2]) - max(a[0], b[0]))
    i_h = abs(min(a[1], b[1]) - max(a[3], b[3]))

    i = i_w * i_h

    a_1 = abs((a[1] - a[0]) * (a[3] - a[2]))
    a_2 = abs((b[1] - b[0]) * (b[3] - b[2]))

    u = a_1 + a_2

    return abs(i / u)

# objectness scores are fucked. figure out how they are implemented in darkent

def inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.log(x / (1 - x))


def intersection(box1, box2):
    x_left = max(box1[1], box2[1])
    y_left = max(box1[2], box2[2])
    x_right = min(box1[1] + box1[3], box2[1] + box2[3])
    y_right = min(box1[2] + box1[4], box2[2] + box2[4])

    return (x_right - x_left) * (y_right - y_left)

# get the cell that the bounding box is in, and its offsets

def get_cell_offsets(label: torch.Tensor) -> tuple:

    center_x = label[1] + torch.div(label[3], 2, rounding_mode='trunc')
    center_y = label[2] + torch.div(label[4], 2, rounding_mode='trunc')
    cell_x = torch.div(center_x, 32, rounding_mode='trunc')
    cell_y = torch.div(center_y, 32, rounding_mode='trunc')
    c_x = center_x - (cell_x * 32)
    c_y = center_y - (cell_y * 32)

    return c_x, c_y, cell_x, cell_y


def get_best_prior(label: torch.Tensor) -> torch.Tensor:
    w, h = label[2], label[3]
    for prior in priors:
        pass

    return NotImplemented


def _readline(f):
    line = f.readline()
    if not line:
        return None
    line = line.replace('\n', '')
    return line
    
# read darknet format cfg file, load into a dictionary later used to build the model
def read_cfg(cfg_file):
    
    model_dict = dict()
    layers = []
    shortcuts = []
    params = None

    normal_layers = [
    'yolo',
    'convolutional', 
    'upsample',
    'route',
    'shortcut'
    ]

    curr_layer = None

    ptr = 0
    
    with open(cfg_file, 'r') as f:
        
        while True:

            line = _readline(f)
            if not line:
                pass
            if line is None:
                break
            line = line.replace('\n', '')
            if line == '' or line[0] == '#':
                pass
            elif line[0] == '[':

                # start a new layer

                curr_layer_name = line[1:-1]
                if curr_layer_name == 'net':
                    curr_layer = dict()
                    curr_layer['name'] = curr_layer_name

                    while line != '':
                        line = _readline(f).split(' ')
                        if line[0] != "#":
                            if line [0] == '':
                                break
                            name = line[0]
                            val = line[2]
                            # test if this is supposed to be a float or not
                            res = '.' in val
                            dig = val.isdigit()
                            if not dig and not res:
                                curr_layer[name] = dig
                            else:
                                val = float(val) if res else int(val)
                            curr_layer[name] = val
                    params = curr_layer
                    print('a')

                elif curr_layer_name in normal_layers:

                    curr_layer = dict()
                    curr_layer['name'] = curr_layer_name

                    while line != '':
                        line = _readline(f)
                        if line is None:
                            break
                        else:
                            line = line.split(' ')

                        if line[0] != "#":
                            if line [0] == '':
                                break
                            name = line[0]
                            val = line[2]
                            if ',' in val and curr_layer_name != 'yolo':
                                val = line[2:]
                                val1 = int(val[0].replace(',', ''))
                                val2 = int(val[1])
                                val = [val1, val2]
                                curr_layer[name] = val
                            else:
                                # test if this is supposed to be a float or not
                                res = '.' in val
                                dig = val.isdigit()
                                if not dig and not res:
                                    curr_layer[name] = dig
                                else:
                                    val = float(val) if res else int(val)
                                curr_layer[name] = val
                    layers.append(curr_layer)
                    
                    ptr += 1
                     
    model_dict['layers'] = layers
    model_dict['params'] = params
    
    return model_dict

# given a config file, create a model
def create_model(config: dict):

    layers = config['layers']
