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

import peaknet_dataset
import random
import math
import os
from peaknet_utils import *
from cfg import parse_cfg
from region_loss import RegionLoss
from darknet import Darknet
from models.tiny_yolo import TinyYoloNet


def validate_batch( model, imgs, labels, json_file=None, mini_batch_size=32, box_size=7, use_cuda=True, writer=None, verbose=False ):
    overall_recall = 0
    
    debug = True

    if json_file is None:
        val_loader = torch.utils.data.DataLoader(
            peaknet_dataset.listDataset(imgs, labels,
                shape=(imgs.shape[2], imgs.shape[3]),
                predict=False,
                box_size=box_size,
                ),
            batch_size=mini_batch_size, shuffle=False)
    else:
        val_loader = torch.utils.data.DataLoader(
            peaknet_dataset.psanaDataset( json_file,
                predict=False,
                box_size=box_size,
                ),
            batch_size=mini_batch_size, shuffle=False)
    
    model.eval()
    region_loss = model.loss
    region_loss.seen = model.seen
    t1 = time.time()
    avg_time = torch.zeros(9)

    val_seen = 0
    for batch_idx, (data, target) in enumerate(val_loader):
        if use_cuda:
            data = data.cuda()
            target= target.cuda()
        data, target = Variable(data), Variable(target)
        output, _= model( data.float() )
        val_seen += data.size(0)
        if debug:
            print("output", output.size())
            print("label length", len(target))
            print("label[0] length", len(target[0]))
        loss, recall = region_loss(output, target)
        overall_recall += data.size(0) * float(recall)
        if writer != None:
            writer.add_scalar('loss_val', loss, model.seen)
            writer.add_scalar('recall_val', recall, model.seen)
            
    overall_recall /= (1.0*val_seen)
    return overall_recall
