import torchvision.transforms as transforms
import numpy as np
import cv2
import torch
from utils.general import read_cfg, non_max_suppression, threshold
from model.yolov3 import YOLOV3
import random
import imgaug.augmenters as iaa
import imgaug as ia
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from utils.params import label_map
import torchvision
from PIL import Image


cfg = read_cfg('cfg/yolov3.cfg')
yolo = YOLOV3(cfg)  # .to('cpu')

# yolo.summary()
yolo.load_weights('weights/yolov3.weights')

img = Image.open('eagle.jpg')

t = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor()
])

img = t(img)

img = img.unsqueeze(0)

yolo.eval()

with torch.no_grad():

    x = yolo(img)

    for i in range(3):
        x[i] = x[i].detach()

    x = torch.cat([x[0], x[1], x[2]], dim=1)

    x = non_max_suppression(x)[0]

    # open the image in cv2

    dets = []
    for item in x:
        print(item)
