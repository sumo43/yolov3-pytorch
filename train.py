from utils.general import read_cfg, non_max_suppression, threshold
from model.yolov3 import YOLOV3

import torch
import cv2
import numpy as np


cfg = read_cfg('cfg/yolov3.cfg')
yolo = YOLOV3(cfg)

yolo.summary()
yolo.load_weights('weights/yolov3.weights')

img = cv2.imread('dog.jpg')
img = cv2.resize(img, (320, 320), cv2.INTER_AREA)
img = np.array(img)
img = np.expand_dims(img, axis=0)
img = np.moveaxis(img, 3, -3)
img = torch.tensor(img).type(torch.float32)

x = yolo(img)

for i in range(3):
    x[i] = x[i].detach()

x = torch.cat([x[0], x[1], x[2]], dim=1)

#x = non_max_suppression(x)

x = threshold(x)
