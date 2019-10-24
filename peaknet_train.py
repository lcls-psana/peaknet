from __future__ import print_function
import sys

import time
import torch
import torch.nn as nn
import torch.nn.init
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
from darknet_utils import get_region_boxes, nms
from darknet import Darknet
#from models.tiny_yolo import TinyYoloNet

def init_model( model ):
    ind = -2
    for block in model.blocks:
        ind = ind + 1
        if block["type"] == "convolutional":
            #print( model.models[ind] )
            torch.nn.init.kaiming_normal( model.models[ind][0].weight )


def updateGrad( model, grad, delta=0, useGPU=False, debug=False ):
    #with torch.no_grad():
    model_dict = dict( model.named_parameters() )
    #model_dict2 = dict( model2.named_parameters() )
    for key, value in model_dict.items():
        #model_dict[key].grad.data = grad[key].data
        model_dict[key]._grad = grad[key]
    model.seen += delta
    if useGPU:
    	model.cuda()
    if debug:
        print("model seen", model.seen)


def optimizer( model, adagrad=False, lr=0.001 ):
    # lr = learning_rate/batch_size
    if adagrad:
        #lr = 0.0005
        decay = 0.005
        optimizer = optim.Adagrad(model.parameters(), lr = lr, weight_decay=decay)
    else:
        #lr = 0.001
        momentum = 0.9
        decay = 0.0005
        optimizer = optim.SGD(model.parameters(), lr=lr,
                            momentum=momentum, dampening=0,
                            weight_decay=decay)
    optimizer.zero_grad()
    return optimizer


def optimize( model, optimizer ):
    optimizer.step()



def adjust_learning_rate(optimizer, batch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = learning_rate
    for i in range(len(steps)):
        scale = scales[i] if i < len(scales) else 1
        if batch >= steps[i]:
            lr = lr * scale
            if batch == steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr/batch_size
    return lr

def train_batch( model, imgs, labels, mini_batch_size=32, box_size=7, use_cuda=True, writer=None, verbose=False ):
#     optimizer = optim.Adagrad(model.parameters(), lr=0.001, weight_decay=0.0005)

    train_loader = torch.utils.data.DataLoader(
        peaknet_dataset.listDataset(imgs, labels,
            shape=(imgs.shape[2], imgs.shape[3]),
            predict=False,
            box_size=box_size,
            ),
        batch_size=mini_batch_size, shuffle=True)

    # lr = adjust_learning_rate(optimizer, processed_batches)
    # logging('epoch %d, processed %d samples, lr %f' % (epoch, epoch * len(train_loader.dataset), lr))
    model.train()
#     model.eval()
    region_loss = model.loss
    region_loss.seen = model.seen
    t1 = time.time()
    avg_time = torch.zeros(9)

    for batch_idx, (data, target) in enumerate(train_loader):
        #print("data min", data.min())
        #print("data max", data.max())
        t2 = time.time()
#         optimizer.zero_grad()
        # adjust_learning_rate(optimizer, processed_batches)
        # processed_batches = processed_batches + 1
        #if (batch_idx+1) % dot_interval == 0:
        #    sys.stdout.write('.')
        #print("timgs type", data.type())
        if verbose:
            for i in range( int(target.size(0) ) ):
                for j in range( int(target.size(1)/5) ):
                    if target[i,j*5+3] < 0.001:
                        break
                    print( i, j,  target[i, j*5+0], target[i, j*5+1], target[i, j*5+2], target[i, j*5+3], target[i, j*5+4] )
        if use_cuda:
            data = data.cuda()
            target= target.cuda()
        t3 = time.time()
        #print( "before", data )
        data, target = Variable(data), Variable(target)
        t4 = time.time()
        t5 = time.time()
        #print( "after", data )
        #output = model( data )
        output, _= model( data )
        #print(output[0,0,:,:])
        #boxes = get_region_boxes(output, conf_thresh, net.model.num_classes, net.model.anchors, net.model.num_anchors)


        #print(output)
        t6 = time.time()
        region_loss.seen = region_loss.seen + data.data.size(0)
        model.seen = region_loss.seen
        #try:
        if verbose:
            print("output", output.size())
            print("label length", len(target))
            print("label[0] length", len(target[0]))
        #region_loss = RegionLoss()
        
        loss, recall = region_loss(output, target)


        if verbose:
            print("label length", len(target))
            print("label[0]", len(target[0]))
            #print("label[0]", target[0,1])
            #print("label[0]", target[0].shape)
            #print("label[1]", target[1].shape)
            #print("label[2]", target[2].shape)
#             raise "something wrong with the labels?"

        t7 = time.time()
        loss.backward()
        t8 = time.time()
#         optimizer.step()
        t9 = time.time()
        if writer != None:
            #writer.add_scalars('loss/recall', {"loss":loss, "recall":recall}, model.seen)
            writer.add_scalar('loss', loss, model.seen)
            writer.add_scalar('recall', recall, model.seen)
            #writer.export_scalars_to_json("./all_scalars.json")
