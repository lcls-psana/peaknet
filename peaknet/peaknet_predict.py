from __future__ import print_function
import sys

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
# from torchvision import datasets, transforms
from torch.autograd import Variable

from peaknet import peaknet_dataset
import random
import math
import os
from peaknet.darknet_utils import *
from peaknet.cfg import parse_cfg
from peaknet.region_loss import RegionLoss
from peaknet.darknet import Darknet
#from models.tiny_yolo import TinyYoloNet


def predict_batch( model, imgs, conf_thresh=0.15, nms_thresh=0.45, batch_size=32, box_size=7, use_cuda=True, writer=None, verbose=False ):

    test_loader = torch.utils.data.DataLoader(
        peaknet_dataset.listDataset(imgs, None,
                        shape=(imgs.shape[2], imgs.shape[3]),
                        predict=True,
                        box_size=box_size,
                        ),
        batch_size=batch_size, shuffle=False)
    #model.train()
    model.eval()

    batch_nms_boxes = []
    
    for batch_idx, data in enumerate(test_loader):
        if verbose:
            print(batch_idx, data.size())
        if use_cuda:
            data = data.cuda()
        data = Variable(data)
        output, _= model( data.float() )
        output = output.data
        #print(output.size())
        #print(output)
        #print(model.num_classes, model.anchors, model.num_anchors)

        boxes = get_region_boxes(output, conf_thresh, model.num_classes, model.anchors, model.num_anchors)
        #print(len(boxes[0][0]))
        nms_boxes = []
        for box in boxes:
            n0 = len(box)
            box = nms(box, nms_thresh)
            n1 = len(box)
            nms_boxes.append( box )
        batch_nms_boxes.append( nms_boxes )
 
    return batch_nms_boxes
