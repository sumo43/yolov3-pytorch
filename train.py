from utils.general import read_cfg, create_model
from model.yolov3 import YOLOV3


cfg = read_cfg('cfg/yolov3.cfg')


yolo = YOLOV3(cfg)
