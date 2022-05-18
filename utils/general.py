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
from utils.params import priors, scales
import math
import time


def coco2yolo(label):
    yolo = torch.clone(label)
    yolo[1] = yolo[1] + (yolo[3] / 2)
    yolo[2] = yolo[2] + (yolo[4] / 2)
    return yolo

# input is (1, 3, n_out, n_grid, n_grid)


"""
arr: y 1, 3, n_out, n_grid...
bounding_box:  select the best prior by iou, 
compare it with the corresponding detections matrix. Save + return the loss
"""


def get_loss_iou(y, bounding_box, scales_index, grid_size):
    bounding_box = coco2yolo(bounding_box)
    # bounding boxes in terms of cells. Should all be 0-10. For x and y, c_x and c_y are their floor
    # grid_size is the size of each box in the grid

    cell_x = torch.floor(bounding_box[0] / 32).type(torch.uint8)
    cell_y = torch.floor(bounding_box[1] / 32).type(torch.uint8)

    print(f'cell_x {cell_x} cell_y {cell_y}')

    # find the prior that has the highest IoU with the bounding box
    # assume that the boxes are centered on top of each other

    best_iou = -1
    best_prior = priors[scales[0][0]]

    cell_x = int(cell_x)
    cell_y = int(cell_y)

    loss = torch.autograd.Variable(torch.tensor(
        0.), type=torch.FloatTensor, requires_grad=True)

    for i, prior_num in enumerate(scales[scales_index]):

        prior = priors[prior_num][0] / \
            grid_size, priors[prior_num][1] / grid_size

        prior_w, prior_h = prior

        iou = compare_iou(bounding_box, prior)

        if(iou > best_iou):
            best_iou = iou
            best_prior = prior
            best_prior_index = i

    x = bounding_box[0]
    y = bounding_box[1]
    w = bounding_box[2]
    h = bounding_box[3]
    _cls = bounding_box[4]

    return loss


def threshold(output: np.array, thres=0.95):
    """threshold the outputs, and convert them to bounding box form

    Args:
        output (np.array): list of yolo head outputs (b_s, x, 85)


    Returns:
        output (np.array): thresholded, smaller list
    """

    bboxes = []
    a = 0

    for i in range(output.shape[1]):
        x, y, w, h, obj = output[:, i, :5].ravel()
        cls_idx = np.argmax(output[:, i, 5:])
        conf = output[:, i, cls_idx + 5]

        if conf > thres:
            a += 1

            print(f'{x} {y} {w} {h} {cls_idx} ')
        else:
            t = list(output[:, i, 5:].ravel())

        bboxes.append((x, y, w, h, conf, cls_idx))
    print((a / output.shape[1]))
    return bboxes


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None):
    """Performs Non-Maximum Suppression(NMS) on inference results
    Returns:
         detections with shape: nx6(x1, y1, x2, y2, conf, cls)
    """

    nc = prediction.shape[2] - 5  # number of classes

    # Settings
    # (pixels) minimum and maximum box width and height
    max_wh = 4096
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 1.0  # seconds to quit after
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    output = [torch.zeros((0, 6), device="cpu")] * prediction.shape[0]

    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[x[..., 4] > conf_thres]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[
                conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            # sort by confidence
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        # Batched NMS
        c = x[:, 5:6] * max_wh  # classes
        # boxes (offset by class), scores
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]

        output[xi] = to_cpu(x[i])

        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output

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


def to_cpu(tensor):
    return tensor.detach().cpu()


def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def compare_iou(bounding_box, prior):

    intersection_w = abs(bounding_box[2] - prior[0])
    intersection_h = abs(bounding_box[3] - prior[1])

    intersection_area = intersection_w * intersection_h

    union_area = (bounding_box[2] * bounding_box[3]) + (prior[0] * prior[1])

    return intersection_area / union_area

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


int_values = [
    'batch',
    'subdivisions',
    'width',
    'height',
    'channels',
    'angle',
    'burn_in',
    'max_batches',
    'batch_normalize',
    'filters',
    'size',
    'stride',
    'pad',
    'classes',
    'num',
    'random',
    'from'
]

float_values = [
    'momentum',
    'decay',
    'saturation',
    'exposure',
    'hue',
    'learning_rate',
    'jitter',
    'ignore_thresh',
    'truth_thresh',
]

string_values = [
    'policy',
    'activation',
]


def get_param(line: str):

    split_line = line.split(' ')
    name = split_line[0]

    if name in int_values:
        value = int(split_line[2])
    elif name in float_values:
        value = float(split_line[2])
    elif name in string_values:
        value = str(split_line[2])
    elif name == 'scales' or name == 'steps':
        value = split_line[2].split(',')
        value = [float(val) for val in value]
    elif name == 'mask':
        value = split_line[2].split(',')
        value = [int(val) for val in value]
    elif name == 'anchors':
        values = split_line[2:]
        value = []

        for val in values:
            if val == '':
                continue

            x, y = val.split(',')[:2]
            value.append((int(x), int(y)))

    elif name == 'layers':
        if len(split_line) == 4:
            value = split_line[2:]
            x = int(value[0].replace(',', ''))
            y = int(value[1])
            value = (x, y)
        else:
            value = int(split_line[2])
    else:
        return None, None

    return name, value


def read_block(infile):

    block = dict()

    # read until you reach the name of a block
    line = _readline(infile)
    while True:
        if line == None:
            return None
        if line == '':
            line = _readline(infile)
            continue
        if line[0] == '[':
            break
        else:
            line = _readline(infile)

    block['name'] = line[1:-1]
    line = _readline(infile)

    # read params until you reach another block name

    prev_line = None

    while True:
        if line == '':
            # skip
            prev_line = infile.tell()
            line = _readline(infile)
            continue
        if line == None or line[0] == '[':
            infile.seek(prev_line)
            break
        else:

            param_name, value = get_param(line)
            if param_name != None:
                block[param_name] = value

            prev_line = infile.tell()
            line = _readline(infile)

    return block

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
        j = 0

        params = None
        layers = []

        while True:

            block = read_block(f)

            if block == None:
                break

            if block['name'] == 'net':
                params = block
            else:
                layers.append(block)

    model_dict['layers'] = layers
    model_dict['params'] = params

    return model_dict
