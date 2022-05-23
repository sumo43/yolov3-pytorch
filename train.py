from model.yolov3 import YOLOV3
from utils.general import read_cfg

im_to_detect = 'samples/street.jpg'

cfg = read_cfg('cfg/yolov3.cfg')
yolo = YOLOV3(cfg)

# yolo.summary()

# finetune on pretrained weights
yolo.load_weights('weights/yolov3.weights')

# when you finetune, the model freezes the ResNet50 weights, and only trains the YOLO parts for faster training
yolo.train('data/val2017', 'data/annotations',
           epochs=15, finetune=True, num_iterations=10)
