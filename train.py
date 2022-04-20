from utils.general import read_cfg, create_model
from model.yolov3 import YOLO


cfg = read_cfg('cfg/yolov3.cfg')


yolo = YOLO(cfg)
