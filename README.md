# yolov3-pytorch

YOLOV3 implementation in PyTorch

### How to use

detect.py:

```
from model.yolov3 import YOLOV3
from utils.general import read_cfg

im_to_detect = 'street.jpg'

cfg = read_cfg('cfg/yolov3.cfg')
yolo = YOLOV3(cfg)

yolo.summary()
yolo.load_weights('weights/yolov3.weights')

dets = yolo.detect(im_to_detect, preview=True, save_img=True)
```

output:

![street_detections.jpg](detections/street_detections.jpg?raw=true)

### TODO

-   [x] Implement weight loading from yolo cfg files
-   [x] Implement detections
-   [ ] Finish loss and backprop
-   [ ] Test out training

### YOLOV3

[Link to Author](https://pjreddie.com/darknet/yolo/)

```
@article{yolov3,
  title={YOLOv3: An Incremental Improvement},
  author={Redmon, Joseph and Farhadi, Ali},
  journal = {arXiv},
  year={2018}
}
```
