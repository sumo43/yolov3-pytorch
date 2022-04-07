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

from utils import inverse_sigmoid, intersection, get_cell_offsets
from utils import YOLOV3
from utils import label_map, priors, scales

print(f'torch cuda: {torch.cuda.is_available()}')

DATA_DIR = os.path.join('data')

# load coco dataset
train_ds = torchvision.datasets.coco.CocoDetection(
    'data/train2017', 'data/annotations/instances_train2017.json')
val_ds = torchvision.datasets.coco.CocoDetection(
    'data/val2017', 'data/annotations/instances_val2017.json')

BASE_PATH = os.path.join('data', 'annotations')
TRAIN_DATA_PATH = os.path.join('data', 'train2017')
VAL_DATA_PATH = os.path.join('data', 'val2017')
TRAIN_CAPTIONS_PATH = os.path.join(BASE_PATH, 'captions_train2017.json')
VAL_CAPTIONS_PATH = os.path.join(BASE_PATH, 'captions_val2017.json')
TRAIN_ANNOTATIONS_PATH = os.path.join(BASE_PATH, 'instances_train2017.json')
VAL_ANNOTATIONS_PATH = os.path.join(BASE_PATH, 'instances_train2017.json')

# we use small dataset for now
num_train_images = 100
num_val_images = 100

transform1 = transforms.ToTensor()


def train(model, loss_fn, optimizer, epochs, train_dl, valid_dl, num_iterations=None):

    batch_size = train_dl.batch_size

    for epoch in range(epochs):

        avg_loss = 0
        acc = 0

        val_avg_loss = 0
        val_acc = 0
        i = 0

        for batch in tqdm(train_dl):

            x, bboxes = batch
            # for now
            bboxes = bboxes[0]
            x = x[0]

            rescale_factor_w = 320 / x.size[0]
            rescale_factor_h = 320 / x.size[1]

            x = x.resize((320, 320))
            x = transform1(x)

            y_pred_1, y_2, y_3 = model(x.unsqueeze(0))
            y_1 = torch.zeros((1, 255, 10, 10))

            box_scale_1 = torch.zeros((1, 255, 10, 10))

            best_prior_choices = []

            for bounding_box in bboxes:

                # scale 1
                # only the best prior is used for each bounding box for each detector scale

                cl, x0, y0, x1, y1 = bounding_box

                # some of these get converted to ints when reading bboxes, which makes following op throw an error
                bounding_box = bounding_box.type(torch.float32)

                # x0, y0, w, h
                bounding_box[1] *= rescale_factor_w
                bounding_box[2] *= rescale_factor_h
                bounding_box[3] *= rescale_factor_w
                bounding_box[4] *= rescale_factor_h

                # xc, yc, w, h
                bounding_box = coco2yolo(bounding_box)

                # bounding boxes in terms of cells. Should all be 0-10. For x and y, c_x and c_y are their floor

                bounding_box[1] /= 32
                bounding_box[2] /= 32
                bounding_box[3] /= 32
                bounding_box[4] /= 32

                c_x = torch.floor(bounding_box[1]).type(torch.uint8)
                c_y = torch.floor(bounding_box[2]).type(torch.uint8)

                cl = bounding_box[0].type(torch.uint8)

                #y_1[c_x][c_y][prior_num * 85]

                # find the prior that has the highest IoU with the bounding box. We only use this prior for loss

                prior_x = c_x + 0.5
                prior_y = c_y + 0.5

                best_iou = 0
                best_prior = priors[scales[0][0]]
                best_prior_index = 0

                i = 0

                c_x = int(c_x)
                c_y = int(c_y)

                for prior_num in scales[0]:
                    prior = priors[prior_num]

                    prior_w = prior[0]
                    prior_h = prior[1]
                    prior_x = c_x + 16
                    prior_y = c_y + 16

                    prior_coords = torch.tensor(
                        (prior_x, prior_y, prior_w, prior_h)) // 32
                    box_coords = bounding_box[1:5]

                    prior_coords = xywh2xyxy(prior_coords)
                    box_coords = xywh2xyxy(box_coords)

                    iou = compare_iou(prior_coords, box_coords)

                    inv_x = inverse_sigmoid(bounding_box[1] - c_x)
                    inv_y = inverse_sigmoid(bounding_box[2] - c_y)
                    inv_w = torch.log(bounding_box[3] / best_prior[0])
                    inv_h = torch.log(bounding_box[4] / best_prior[1])
                    inv_o = iou

                    print(inv_x, inv_y, inv_w, inv_h, inv_o)

                    #pred_x = torch.sigmoid(y_1[0][best_prior_index * 85][c_x][c_y]) + c_x
                    #pred_y = torch.sigmoid(y_1[0][best_prior_index * 85 + 1][c_x][c_y]) + c_y
                    #pred_w = best_prior[0] * torch.exp(y_1[0][best_prior_index * 85 + 2][c_x][c_y]) / 32
                    #pred_h = best_prior[1] * torch.exp(y_1[0][best_prior_index * 85 + 3][c_x][c_y]) / 32

                    #preds = torch.tensor((pred_x, pred_y, pred_w, pred_h))

                    #pred_xyxy = xywh2xyxy(preds)

                    #gt_xyxy = xywh2xyxy(torch.tensor(bounding_box[1:5]))
                    pred_o = iou

                    y_1[0][i * 85][c_x][c_y] = inv_x
                    y_1[0][i * 85 + 1][c_x][c_y] = inv_y
                    y_1[0][i * 85 + 2][c_x][c_y] = inv_w
                    y_1[0][i * 85 + 3][c_x][c_y] = inv_h
                    y_1[0][i * 85 + 4][c_x][c_y] = iou
                    ind = i * 85 + 5 + cl
                    y_1[0, ind, c_x, c_y] = 1

                    i += 1

            optimizer.zero_grad()
            loss = loss_fn(y_pred_1, y_1)
            loss.backward()
            optimizer.step()

        avg_loss /= (len(train_dl) * batch_size)
        #acc /= (len(train_dl) * batch_size)

        val_avg_loss /= (len(valid_dl) * batch_size)
        #val_acc /= (len(valid_dl) * batch_size)

        print(
            f'epoch: {epoch} loss: {avg_loss} acc: {acc} val_loss: {val_avg_loss} val_acc: {val_acc}')
