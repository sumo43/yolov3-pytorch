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

img = Image.open('messi.jpg')

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
    cv_im = cv2.imread('messi.jpg')

    for det in x:
        x0, y0, x1, y1, conf, _cls = det

        cv_im = cv2.resize(cv_im, (320, 320))

        conf = round(float(conf), 4)

        cv2.rectangle(cv_im, (x0, y0), (x1, y1),
                      (0, 255, 0), 2)
        cv2.putText(cv_im, f'{label_map[_cls]} {conf}', (x0, y0 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

    cv2.imshow('eagle', cv_im)
    cv2.waitKey(0)

    dets = []
    for item in x:
        print(item)
