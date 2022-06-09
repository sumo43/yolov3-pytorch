from model.yolov3 import YOLOV3
from utils.general import read_cfg
import torch 

im_to_detect = 'samples/street.jpg'

cfg = read_cfg('cfg/yolov3.cfg')
yolo = YOLOV3(cfg)

yolo.load_state_dict(torch.load('weights/saved_model.pt'))

yolo.summary()
#yolo.load_weights('weights/yolov3.weights')

dets = yolo.detect(im_to_detect, preview=False, save_img=True)

yolo.freeze()
yolo.unfreeze()
