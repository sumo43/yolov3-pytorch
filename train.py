from utils.general import read_cfg, threshold
from model.yolov3 import YOLOV3

import torch
import cv2


cfg = read_cfg('cfg/yolov3.cfg')
yolo = YOLOV3(cfg)

yolo.summary()
yolo.load_weights('weights/yolov3.weights')

img = cv2.imread('messi.jpg')
img = cv2.resize(img, (320, 320), cv2.INTER_AREA)
img = torch.tensor(img).permute(2, 0, 1)
img = img.reshape(
    1, 3, 320, 320).type(torch.float32)

img /= 255.0

x = yolo(img)
#x = threshold(x)
