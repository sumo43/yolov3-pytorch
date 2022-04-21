from utils.general import read_cfg, create_model
from model.yolov3 import YOLOV3
import torch


cfg = read_cfg('cfg/yolov3.cfg')


yolo = YOLOV3(cfg)


x = torch.zeros((1, 3, 320, 320))
yolo(x)
