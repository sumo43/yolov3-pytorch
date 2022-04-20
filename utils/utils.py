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
from utils.params import label_map, priors, scales

import math

def compare_iou(a, b):
    i = abs(a[2] - b[2]) * abs(a[3] - b[3])
    u = abs(a[2] * a[3]) + abs(b[2] * b[3])
    return i / u

def build_groundtruth(arr, bounding_box, scales_index, grid_size):
    bounding_box = coco2yolo(bounding_box)
    # bounding boxes in terms of cells. Should all be 0-10. For x and y, c_x and c_y are their floor
                
    bounding_box[1] /= grid_size
    bounding_box[2] /= grid_size
    bounding_box[3] /= grid_size
    bounding_box[4] /= grid_size

    c_x = torch.floor(bounding_box[1]).type(torch.uint8)
    c_y = torch.floor(bounding_box[2]).type(torch.uint8)
                
    cl = bounding_box[0].type(torch.uint8)
    #y_1[c_x][c_y][prior_num * 85]
                
    # find the prior that has the highest IoU with the bounding box. We only use this prior for loss
    best_iou = -1
    best_prior = priors[scales[0][0]]
                
    i = 0
        
    c_x = int(c_x)
    c_y = int(c_y)
                
    for prior_num in scales[scales_index]:
        prior = priors[prior_num][0] / grid_size, priors[prior_num][1] / grid_size

        prior_w, prior_h = prior

        prior_coords = torch.tensor((bounding_box[1], bounding_box[2], prior_w, prior_h))
        box_coords = bounding_box[1:5]

        iou = compare_iou(prior_coords, box_coords)

        if(iou > best_iou):
            best_iou = iou
            best_prior = prior
            best_prior_index = i

        i += 1
                    
    inv_x = inverse_sigmoid(bounding_box[1] - c_x)
    inv_y = inverse_sigmoid(bounding_box[2] - c_y)
    inv_w = torch.log(bounding_box[3] / best_prior[0])
    inv_h = torch.log(bounding_box[4] / best_prior[1])
    inv_o = iou

    arr[0][best_prior_index * 85][c_x][c_y] = inv_x
    arr[0][best_prior_index * 85 + 1][c_x][c_y] = inv_y
    arr[0][best_prior_index * 85 + 2][c_x][c_y] = inv_w
    arr[0][best_prior_index * 85 + 3][c_x][c_y] = inv_h
    arr[0][best_prior_index * 85 + 4][c_x][c_y] = iou
    """
    if(cl < 80):
        arr[0][best_prior_index * 85 + (5 + int(cl))][c_x][c_y] = 1 
    """
    
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


def inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.log(x / (1 - x))

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
    net = None

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
                    net = curr_layer

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
                elif current_layer_name in ['shortcut', 'route']:
                    
