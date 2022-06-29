# yolov3-pytorch

YOLOV3 implementation in PyTorch. This repo borrows heavily from https://github.com/eriklindernoren/PyTorch-YOLOv3. Not really intended to be used for anything other than educational purposes.

### How to use

detect.py:

```python
from model.yolov3 import YOLOV3
from utils.general import read_cfg

im_to_detect = 'samples/street.jpg'

cfg = read_cfg('cfg/yolov3.cfg')
yolo = YOLOV3(cfg)

yolo.summary()
yolo.load_weights('weights/yolov3.weights')

dets = yolo.detect(im_to_detect, preview=True, save_img=True)
```

output:

![street_detections.jpg](detections/street_detections.jpg?raw=true)

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
