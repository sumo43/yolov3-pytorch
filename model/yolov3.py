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


class YOLOV3(nn.Module):
    def __init__(self):
        # input size = (256, 256)
        super(YOLOV3, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, padding=(1, 1))
        # downsample
        self.conv2 = nn.Conv2d(32, 64, 3, stride=(2, 2), padding=(1, 1))
        self.conv3 = ResidualBlock(32, 64)

        self.conv4 = nn.Conv2d(64, 128, 3, stride=(2, 2), padding=(1, 1))
        self.conv5 = torch.nn.Sequential(ResidualBlock(64, 128),
                                         ResidualBlock(64, 128))

        self.conv6 = nn.Conv2d(128, 256, 3, stride=(2, 2), padding=(1, 1))
        self.conv7 = torch.nn.Sequential(ResidualBlock(128, 256),
                                         ResidualBlock(128, 256),
                                         ResidualBlock(128, 256),
                                         ResidualBlock(128, 256),
                                         ResidualBlock(128, 256),
                                         ResidualBlock(128, 256),
                                         ResidualBlock(128, 256),
                                         ResidualBlock(128, 256))

        self.conv8 = nn.Conv2d(256, 512, 3, stride=(2, 2), padding=(1, 1))
        self.conv9 = torch.nn.Sequential(ResidualBlock(256, 512),
                                         ResidualBlock(256, 512),
                                         ResidualBlock(256, 512),
                                         ResidualBlock(256, 512),
                                         ResidualBlock(256, 512),
                                         ResidualBlock(256, 512),
                                         ResidualBlock(256, 512),
                                         ResidualBlock(256, 512))

        self.conv10 = nn.Conv2d(512, 1024, 3, stride=(2, 2), padding=(1, 1))
        self.conv11 = torch.nn.Sequential(ResidualBlock(512, 1024),
                                          ResidualBlock(512, 1024),
                                          ResidualBlock(512, 1024),
                                          ResidualBlock(512, 1024))

        self.conv12 = nn.Conv2d(1024, 512, 1)
        self.conv13 = nn.Conv2d(512, 1024, 3, padding=(1, 1))
        self.conv14 = nn.Conv2d(1024, 512, 1)
        self.conv15 = nn.Conv2d(512, 1024, 3, padding=(1, 1))
        self.conv16 = nn.Conv2d(1024, 512, 1)
        self.conv17 = nn.Conv2d(512, 1024, 3, padding=(1, 1))

        # first yolo output
        self.conv18 = torch.nn.Conv2d(1024, 255, 1)

        self.conv19 = torch.nn.Conv2d(512, 256, 1)

        self.conv20 = nn.Conv2d(768, 256, 1)
        self.conv21 = nn.Conv2d(256, 512, 3, padding=(1, 1))
        self.conv22 = nn.Conv2d(512, 256, 1)
        self.conv23 = nn.Conv2d(256, 512, 3, padding=(1, 1))
        self.conv24 = nn.Conv2d(512, 256, 1)
        self.conv25 = nn.Conv2d(256, 512, 3, padding=(1, 1))

        self.conv26 = nn.Conv2d(512, 255, 1)

        self.conv27 = nn.Conv2d(256, 128, 1)

        self.conv28 = nn.Conv2d(384, 128, 1)
        self.conv29 = nn.Conv2d(128, 256, 3, padding=(1, 1))
        self.conv30 = nn.Conv2d(256, 128, 1)
        self.conv31 = nn.Conv2d(128, 256, 3, padding=(1, 1))
        self.conv32 = nn.Conv2d(256, 128, 1)
        self.conv33 = nn.Conv2d(128, 256, 3, padding=(1, 1))

        self.conv34 = nn.Conv2d(256, 255, 1)

        self.upsample = torch.nn.Upsample(scale_factor=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x_route36 = x
        x = self.conv8(x)
        x = self.conv9(x)
        x_route61 = x

        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.conv14(x)
        x = self.conv15(x)
        x = self.conv16(x)
        x_route79 = x
        x = self.conv17(x)
        x = self.conv18(x)
        yolo_output_1 = x
        x = x_route79
        x = self.conv19(x)
        x = self.upsample(x)
        x = torch.cat((x, x_route61), dim=1)

        x = self.conv20(x)
        x = self.conv21(x)
        x = self.conv22(x)
        x = self.conv23(x)
        x = self.conv24(x)
        x_route91 = x
        x = self.conv25(x)

        x = self.conv26(x)
        yolo_output_2 = x

        x = x_route91
        x = self.conv27(x)
        x = self.upsample(x)
        x = torch.cat((x, x_route36), dim=1)

        x = self.conv28(x)
        x = self.conv29(x)
        x = self.conv30(x)
        x = self.conv31(x)
        x = self.conv32(x)
        x = self.conv33(x)

        x = self.conv34(x)

        yolo_output_3 = x

        return yolo_output_1, yolo_output_2, yolo_output_3

    DATA_DIR = os.path.join('data')
