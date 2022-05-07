from utils.general import read_cfg
from model.yolov3 import YOLOV3
import torch


cfg = read_cfg('cfg/yolov3.cfg')
#yolo = YOLOV3(cfg)

# yolo.summary()
# yolo.load_weights('weights/yolov3.weights')
#x = torch.zeros((1, 3, 320, 320))

#x = yolo(x)

#print([y.shape for y in x])
