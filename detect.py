from model.yolov3 import YOLOV3
from utils.general import read_cfg

im_to_detect = 'samples/street.jpg'

cfg = read_cfg('cfg/yolov3.cfg')
yolo = YOLOV3(cfg)

yolo.summary()
yolo.load_weights('weights/yolov3.weights')

dets = yolo.detect(im_to_detect, preview=True, save_img=True)
