import torch
import torch.nn as nn
import os

import numpy as np
import os

from PIL import Image
from torchvision import transforms
from utils.general import non_max_suppression
from utils.params import label_map
import cv2


def generate_conv(layer: dict, in_channels, bias=False):

    filters = layer['filters']
    stride = layer['stride']
    pad = layer['pad']
    kernel_size = layer['size']
    activation = layer['activation']

    if kernel_size == 1:
        pad = 0

    conv = nn.Conv2d(in_channels, filters,
                     kernel_size=kernel_size, stride=stride, padding=pad, bias=bias)

    return conv


class YoloHead(nn.Module):
    def __init__(self, info: dict):

        super(YoloHead, self).__init__()

        self.anchors = [info['anchors'][i] for i in info['mask']]
        self.num_anchors = 3
        self.num_classes = 80
        self.no = self.num_classes + 5  # number of outputs per anchor
        self.grid = torch.zeros(1)  # TODO

    # some code borrowed from https://github.com/eriklindernoren/PyTorch-YOLOv3
    def forward(self, x, img_size=320):

        stride = img_size // x.shape[2]
        self.stride = stride
        bs, _, ny, nx = x.shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
        x = x.view(bs, 3, 85, ny,
                   nx).permute(0, 1, 3, 4, 2).contiguous()

        anchor_mask = torch.ones((1, 3, ny, nx, 2))
        for i in range(3):
            anchor_mask[:, i, :, :, 0] = self.anchors[i][0]
            anchor_mask[:, i, :, :, 1] = self.anchors[i][1]

        self.grid = self._make_grid(nx, ny)

        x[..., 0:2] = (x[..., 0:2].sigmoid() + self.grid) * stride  # xy
        x[..., 2:4] = torch.exp(x[..., 2:4]) * anchor_mask  # wh
        x[..., 4:] = x[..., 4:].sigmoid()
        x = x.view(bs, -1, 85)

        return x

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid(
            [torch.arange(ny), torch.arange(nx)], indexing='ij')
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class YoloRoute(nn.Module):
    def __init__(self):

        super(YoloRoute, self).__init__()

    def forward(self, x):
        return x


class YoloShortcut(nn.Module):
    def __init__(self):

        super(YoloShortcut, self).__init__()

    def forward(self, x):
        return x


"""

def read_config(cfg: dict):
    _summary_

   Args:
        cfg(dict): config input from read_cfg

    Returns:
        layers: a list of PyTorch layers representing the model. Is converted later to a torch.nn.Sequential

    i = 1
    l = 0

    layers = []
    im_channels = 3
    prev_conv_inc = None

    self.process_outputs = False

    self.shortcuts = dict()
    saved_x = dict()

    self.routes = dict()
    self.single_routes = dict()

    for layer in cfg['layers']:
        if layer['name'] == 'convolutional':

            bias = True

            if 'batch_normalize' in layer.keys():
                bias = False

            if prev_conv_inc == None:
                conv_layer = generate_conv(layer, im_channels, bias=bias)
            else:
                conv_layer = generate_conv(layer, prev_conv_inc, bias=bias)

            curr_layer = []
            curr_layer.append(conv_layer)

            prev_conv_inc = layer['filters']

            if 'batch_normalize' in layer.keys() and layer['batch_normalize'] == 1:
                bn_layer = nn.BatchNorm2d(layer['filters'])
                curr_layer.append(bn_layer)

            if layer['activation'] == 'leaky':
                relu_layer = nn.LeakyReLU()
                curr_layer.append(relu_layer)
                no_bias = False
            elif layer['activation'] == 'relu':
                relu_layer = nn.ReLU()
                curr_layer.append(relu_layer)

            curr_layer = nn.Sequential(*curr_layer)

            layers.append(curr_layer)
            i += 1
            l += 1

        elif layer['name'] == 'upsample':
            upsample_layer = torch.nn.Upsample(scale_factor=2)
            layers.append(upsample_layer)
            i += 1
            l += 1

        elif layer['name'] == 'shortcut':
            _from = int(layer['from'])
            if _from < 1:
                _from = i + _from
            _to = i

            self.shortcuts[_from] = _to
            layers.append(YoloShortcut())
            i += 1

        elif layer['name'] == 'route':

            # the route has 2 layers, we concatenate
            if isinstance(layer['layers'], list):
                _from = int(layer['layers'][0])
                if _from < 1:
                    _from = i + _from
                _to = int(layer['layers'][1])

                self.routes[_to] = _from + 2

            else:
                _from = int(layer['layers'])
                if _from < 1:
                    _from = i + _from
                _to = i

                self.single_routes[_from] = _to + 2

            layers.append(YoloRoute())
            i += 1

        elif layer['name'] == 'yolo':

            layers.append(YoloHead())
            self.yolo_layers.append(i)
            i += 1
            l += 1

    for i in self.routes.keys():
        _from = self.routes[i]
        conv_shape_in = layers[i - 1][0].weight.shape[1] + \
            layers[_from - 2][0].weight.shape[0] * 2
        conv_shape_out = layers[_from - 2][0].weight.shape[0]
        layers[_from + 1][0] = nn.Conv2d(conv_shape_in,
                                         conv_shape_out, 1, 1, bias=False)

    for i in self.single_routes.keys():

        _from = self.single_routes[i]
        conv_shape_in = layers[i][0].weight.shape[1]
        conv_shape_out = layers[_from][0].weight.shape[0]
        layers[_from][0] = nn.Conv2d(
            conv_shape_in, conv_shape_out, 1, 1, bias=False)

    return layers
"""


class YOLOV3(nn.Module):
    def __init__(self, cfg):
        # input size = (256, 256)
        super(YOLOV3, self).__init__()

        self.training = False

        # get model metaparameters from cfg
        params = cfg['params']

        batch_size, subdivs, width, height, channels = int(params['batch']), int(
            params['subdivisions']), int(params['width']), int(params['height']), int(params['channels'])
        momentum, decay = float(params['momentum']), float(params['decay'])

        cfg_layers = cfg['layers']

        self.yolo_layers = []

        i = 1
        l = 0

        layers = []
        im_channels = 3
        prev_conv_inc = None

        self.process_outputs = False

        self.shortcuts = dict()
        saved_x = dict()

        self.routes = dict()
        self.single_routes = dict()

        for layer in cfg['layers']:
            if layer['name'] == 'convolutional':

                bias = True

                if 'batch_normalize' in layer.keys():
                    bias = False

                if prev_conv_inc == None:
                    conv_layer = generate_conv(layer, im_channels, bias=bias)
                else:
                    conv_layer = generate_conv(layer, prev_conv_inc, bias=bias)

                curr_layer = []
                curr_layer.append(conv_layer)

                prev_conv_inc = layer['filters']

                if 'batch_normalize' in layer.keys() and layer['batch_normalize'] == 1:
                    bn_layer = nn.BatchNorm2d(layer['filters'])
                    curr_layer.append(bn_layer)

                if layer['activation'] == 'leaky':
                    relu_layer = nn.LeakyReLU(0.1)
                    curr_layer.append(relu_layer)
                    no_bias = False
                elif layer['activation'] == 'relu':
                    relu_layer = nn.ReLU()
                    curr_layer.append(relu_layer)

                curr_layer = nn.Sequential(*curr_layer)

                layers.append(curr_layer)
                i += 1
                l += 1

            elif layer['name'] == 'yolo':

                layers.append(YoloHead(layer))
                self.yolo_layers.append(i - 1)

                i += 1
                l += 1

            elif layer['name'] == 'upsample':
                upsample_layer = torch.nn.Upsample(scale_factor=2)
                layers.append(upsample_layer)
                i += 1
                l += 1

            elif layer['name'] == 'shortcut':
                _from = int(layer['from'])
                if _from < 1:
                    _from = i + _from
                _to = i
                _to -= 1

                _from -= 1

                self.shortcuts[_from] = _to
                layers.append(YoloShortcut())
                i += 1

            elif layer['name'] == 'route':

                # the route has 2 layers, we concatenate
                if isinstance(layer['layers'], tuple):
                    _from = int(layer['layers'][0])
                    if _from < 1:
                        _from = i + _from

                    _to = int(layer['layers'][1])

                    self.routes[_to] = _from

                else:
                    _from = int(layer['layers'])
                    if _from < 1:
                        _from = i + _from
                    _to = i

                    _from -= 1
                    _to -= 1

                    self.single_routes[_from] = _to

                layers.append(YoloRoute())
                i += 1

        for i in self.routes.keys():
            _from = self.routes[i]
            _next = _from + 1

            conv_shape_in = layers[i - 1][0].weight.shape[0] + \
                (layers[_next][0].weight.shape[0])

            conv_shape_out = layers[_next][0].weight.shape[0]
            layers[_next][0] = nn.Conv2d(conv_shape_in,
                                         conv_shape_out, 1, 1, bias=False)

        for i in self.single_routes.keys():
            _to = self.single_routes[i]
            conv_shape_in = layers[i][0].weight.shape[0]
            conv_shape_out = layers[_to + 1][0].weight.shape[0]
            layers[_to + 1][0] = nn.Conv2d(
                conv_shape_in, conv_shape_out, 1, 1, bias=False)

        self.layers = nn.Sequential(*layers)

    def summary(self):
        for i, layer in enumerate(self.layers):
            print(f'{i}: {layer}')

    def getexp(self, param):
        return param.split('.')[:3]

    def load_weights(self, weights_file):

        ptr = 0
        i = 0
        even = 0
        prev = []

        sd = self.state_dict()
        sd_keys = list(sd.keys())

        len_sd = len(sd_keys)

        with open(weights_file, 'rb') as f:
            header = np.fromfile(f, dtype=np.int32, count=5)
            weights = np.fromfile(f, dtype=np.float32)

        ptr = 0

        for layer in self.layers:

            layer_type = type(layer)

            if isinstance(layer, torch.nn.modules.container.Sequential):

                # batch norm OR conv bias
                # then weights

                conv = layer[0]

                if len(layer) > 1:

                    bn = layer[1]
                    bs = bn.bias.data.shape.numel()

                    bn.bias.data = torch.from_numpy(weights[ptr:ptr+bs])
                    ptr += bs

                    # beta (weight)
                    bn.weight.data = torch.from_numpy(weights[ptr:ptr+bs])
                    ptr += bs
                    # running mean
                    bn.running_mean.data = torch.from_numpy(
                        weights[ptr:ptr+bs])
                    ptr += bs
                    # running variance
                    bn.running_var.data = torch.from_numpy(weights[ptr:ptr+bs])
                    ptr += bs
                    # sd[param_num_batches_tracked] =  torch.from_numpy(weights[ptr:ptr+bs])
                    # ptr += bs
                else:
                    bs = conv.bias.data.shape.numel()
                    conv.bias.data = torch.from_numpy(
                        weights[ptr:ptr+bs])
                    ptr += bs

                ws = tuple(conv.weight.data.shape)
                wsz = np.prod(ws)
                # beta (weight)
                conv.weight.data = torch.from_numpy(
                    weights[ptr:ptr+wsz]).view_as(conv.weight.data)
                ptr += wsz

        print(f'weights loaded: {ptr}')

    def forward(self, x):

        saved_x_shortcuts = dict()
        saved_x_routes = dict()
        saved_single_x_routes = dict()
        yolo_outputs = []

        shortcuts = 0
        routes = 0
        single_routes = 0
        routes = 0

        for i, layer in enumerate(self.layers):

            if type(layer) == torch.nn.Sequential or type(layer) == torch.nn.Upsample:
                x = layer(x)

            if i in self.single_routes:
                _to = self.single_routes[i]
                saved_single_x_routes[_to] = x

            if i in saved_single_x_routes.keys():
                shape = x.shape
                x = saved_single_x_routes[i]

            if i in saved_x_shortcuts.keys():
                x = x + saved_x_shortcuts[i]
                shape = x.shape
            if i in self.shortcuts:
                _to = self.shortcuts[i]
                saved_x_shortcuts[_to] = x

            if i in self.routes:
                _to = self.routes[i]
                saved_x_routes[_to] = x
            elif i in saved_x_routes.keys():
                x = torch.cat((x, saved_x_routes[i]), 1)
                shape = x.shape

            if i in self.yolo_layers:
                x = layer(x)
                yolo_outputs.append(x)

        return yolo_outputs

    def detect(self, img_name, preview=False, save_img=False):
        """detect objects in image, return a list of detections. Optionally also show a cv2 preview of the image and/or save the image with bounding boxes applied.

        Args:
            img (_type_): _description_
            preview (bool, optional): _description_. Defaults to False.
            save_img (bool, optional): _description_. Defaults to False.
        """        ""

        img = Image.open(img_name)

        t = transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.ToTensor()
        ])

        img = t(img)

        img = img.unsqueeze(0)

        # yolo.detect('messi.jpg', preview=True, save_img=True)

        self.eval()

        with torch.no_grad():

            x = self(img)

            for i in range(3):
                x[i] = x[i].detach()

            x = torch.cat([x[0], x[1], x[2]], dim=1)

            x = non_max_suppression(x, 0.60, 0.6)[0]

            # open the image in cv2
            cv_im = cv2.imread(img_name)

            if preview or save_img:

                for det in x:
                    x0, y0, x1, y1, conf, _cls = det

                    cv_im = cv2.resize(cv_im, (320, 320))

                    conf = round(float(conf), 4)

                    x0 = int(x0)
                    y0 = int(y0)
                    x1 = int(x1)
                    y1 = int(y1)
                    _cls = int(_cls)

                    # (x1, y1), (0, 255, 0), 2)
                    cv2.rectangle(cv_im, (x0, y0), (x1, y1), (0, 255, 0), 1)
                    cv2.putText(cv_im, f'{label_map[_cls]} {conf}', (x0, y0 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

                if preview:
                    cv2.imshow('(tap any key to destroy)', cv_im)
                    cv2.waitKey(0)
                if save_img:
                    if 'jpg' in img_name:
                        new_name = img_name.replace('.jpg', '_detections.jpg')
                    elif 'jpeg' in img_name:
                        new_name = img_name.replace(
                            '.jpeg', '_detections.jpeg')
                    cv2.imwrite(os.path.join(
                        os.getcwd(), 'detections', new_name), cv_im)

            return x
