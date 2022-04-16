## YOLO notes


### YOLO Archiecture

Inputs
- Image in multiple of 32 (416x416, 320x320, etc)
- Set of bounding boxes [...(x0, y0, w, h, conf, cls)]
- Set of anchor masks (Can just get from Darknet YOLO, or make your own with cluster centroids)
  - The width and height outputs of the model will be the offsets from the anchor masks

YOLO Heads
- Large grid (32x32 boxes), medium grid (16x16), small grid(8x8)
- Train jointly

Detections Format
  - A = number of anchors per head. The official implementation uses 3
  - C = Number of classes
  - G = Number of boxes per head (The large head outputs 10 boxes for a 320x320 image)
  - B = batch dimension
  - The yolo head output is in the format (B, A * (C + 5), G, G)
  - For example, for a 320x320 image and a model trained on COCO (80 classes), the output is (B, 255, 10, 10)

Image Processing Pipeline
- Normalize to values in between -1, 1 (TODO move into model forward() so this is done always)
- Divide bounding boxes by the grid size for each anchor, so that they are relative to the anchors

















































