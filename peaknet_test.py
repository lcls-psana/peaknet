from __future__ import print_function
import sys

import time
import torch
import torch.nn as nn
#import torch.nn.init
import torch.nn.functional as F
#import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from torch.autograd import Variable

import peaknet_dataset
import random
import math
import os
from utils import *
from cfg import parse_cfg
from region_loss import RegionLoss
from darknet_utils import get_region_boxes, nms
from darknet import Darknet
#from models.tiny_yolo import TinyYoloNet


def test_batch( model, imgs, labels, batch_size=1, box_size=7, use_cuda=True, writer=None ):
    debug = False
    
    data_loader = torch.utils.data.DataLoader(
        peaknet_dataset.listDataset(imgs, labels,
                        shape=(imgs.shape[2], imgs.shape[3]),
                        shuffle=False,
                        train=False,
                        box_size=box_size,
                        batch_size=batch_size
                        ),
        batch_size=batch_size, shuffle=False)

    model.eval()

    t1 = time.time()

    for batch_idx, (data, target) in enumerate(train_loader):
        t2 = time.time()
       
        if use_cuda:
            data = data.cuda()
            target= target.cuda()

        t3 = time.time()
        data, target = Variable(data), Variable(target)
        t4 = time.time()
        output, _= model( data )
        t6 = time.time()
     
        loss, recall = region_loss(output, target)
      
        t7 = time.time()
       
        if writer != None:
            writer.add_scalar('dev-loss', loss, model.seen)
	    writer.add_scalar('dev-recall', recall, model.seen) 	
       
