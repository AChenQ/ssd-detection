import logging
import os
from collections import OrderedDict

import cv2
import numpy as np
import requests
import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm

from .detection import Detect
from .ssd_net_vgg import SSD
from .Config import class_num
from .utils import default_prior_box
from .voc0712 import VOC_CLASSES


def download(url, file_path):
    headers = {}

    r1 = requests.get(url, stream=True)
    total_size = int(r1.headers["Content-Length"])

    if os.path.exists(file_path):
        temp_size = os.path.getsize(file_path)
    else:
        temp_size = 0
    headers = {"Range": "bytes=%d-" % temp_size, "Connection": "keep-alive"}

    r = requests.get(url, stream=True, verify=False, headers=headers)

    with tqdm(total=total_size, unit="B", unit_scale=True, unit_divisor=1024) as pbar:
        pbar.update(temp_size)
        with open(file_path, "ab") as f:
            for chunk in r.iter_content(chunk_size=1024):
                pbar.update(1024)
                if chunk:
                    temp_size += len(chunk)
                    f.write(chunk)
                    f.flush()


class Predictor:
    def __init__(self):
        """
        initializer
        """

        self.path = os.path.abspath(os.path.dirname(__file__))

        logging.info("===============start init ssd===============")
        ssd_path = os.path.join(self.path, "weights/ssd300_VOC_120000.pth")
        if not os.path.exists(os.path.join(self.path, "weights")):
            os.mkdir(os.path.join(self.path, "weights"))
        if not os.path.exists(ssd_path):
            download(
                "https://algorithm-graviti.oss-cn-shanghai.aliyuncs.com/tmp/ssd300_VOC_120000.pth",
                ssd_path,
            )

        state_dict = torch.load(ssd_path, map_location="cpu")
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v

        self.net = SSD()
        self.net.load_state_dict(new_state_dict)

    def predict(self, img_data: bytes) -> dict:
        """
        Do the detection job
        :param img_data: the binary data of one image file
        :return: the detect result
        """
        # load image data to open_cv
        buf = np.fromstring(img_data, np.uint8)
        image = cv2.imdecode(buf, cv2.IMREAD_COLOR)

        x = cv2.resize(image, (300, 300)).astype(np.float32)
        x -= (104.0, 117.0, 123.0)
        x = x.astype(np.float32)
        x = x[:, :, ::-1].copy()
        x = torch.from_numpy(x).permute(2, 0, 1)
        xx = Variable(x.unsqueeze(0))
        y = self.net(xx)

        softmax = nn.Softmax(dim=-1)
        detect = Detect(class_num, 0, 200, 0.01, 0.45)
        priors = default_prior_box()

        loc, conf = y
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        detections = detect.forward(
            loc.view(loc.size(0), -1, 4),
            softmax(conf.view(conf.size(0), -1, class_num)),
            torch.cat([o.view(-1, 4) for o in priors], 0),
        ).data

        labels = VOC_CLASSES

        scale = torch.Tensor(image.shape[1::-1]).repeat(2)
        data = []
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= 0.6:
                score = detections[0, i, j, 0]
                category = labels[i]
                pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                data.append(
                    {
                        "category": category,
                        "attributes": {"confidence": float(score.cpu().numpy())},
                        "box2d": {
                            "xmin": float(pt[0]),
                            "ymin": float(pt[1]),
                            "xmax": float(pt[2]),
                            "ymax": float(pt[3]),
                        },
                    }
                )
                j += 1
        return data

    def teardown(self) -> None:
        return
