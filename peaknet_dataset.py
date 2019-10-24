#!/usr/bin/python
# encoding: utf-8

import os, sys
import random
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.autograd import Variable
#from PIL import Image
sys.path.append(os.path.abspath('../pytorch-yolo3'))
from darknet_utils import read_truths_args, read_truths
#from image import *
from peaknet_utils import json_parser, psana_img_loader, psanaRun


class psanaDataset(Dataset):
    def __init__(self, json_file, shape=None, shuffle=False, predict=False,
                    box_size=7, n_panels=32):
       self.df = json_parser(json_file, mode="valid")
       self.nSamples  = 32 * len(self.df)
       self.predict = predict
       self.shape = shape
       self.box_size = box_size
       self.psana_runs = {}

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        
        m = 32
        
        ind1 = index / m
        ind2 = index % m
        
        
        exp = str(self.df.loc[ind1,"exp"])
        run = str(self.df.loc[ind1,"run"])
        det = str(self.df.loc[ind1,"detector"])
        
        if (exp, run, det) in self.psana_runs:
            (this_run, det, times) = self.psana_runs[(exp, run, det)]
        else:
            (this_run, detector, times) = psanaRun(exp, run, det)
            self.psana_runs[(exp, run, det)] = (this_run, detector, times)
            
        event_idx = int(self.df.loc[ind1,"event"]) 
            
        evt = this_run.event(times[event_idx])
        calib = detector.calib(evt) * detector.mask(evt, calib=True, status=True,
                              edges=True, central=True,
                              unbond=True, unbondnbrs=True)
        imgs = calib
        
        maxPeaks = 1024
        (m,h,w) = imgs.shape
        


        new_h = 192
        new_w = 392
        timg = torch.zeros( ( 1, new_h, new_w) )
        #img = imgs[ind2,:,:] * 255.0/ 15000.0
        img = 1.0 * imgs[ind2,:,:] / np.max(imgs[ind2,:,:])
        timg[0,4:189,2:390] = torch.from_numpy( img )
        timg = timg.view(-1, new_h, new_w )

        if self.predict:
            return timg
        else:
#           #TODO: labels = psana_labels()
            r = np.reshape( self.labels[ind1][2][ self.labels[ind1][1]==ind2 ], (-1,1) )
            c = np.reshape( self.labels[ind1][3][ self.labels[ind1][1]==ind2 ], (-1,1) )
            bh = np.reshape( self.labels[ind1][4][ self.labels[ind1][1]==ind2 ], (-1,1) )
            bw = np.reshape( self.labels[ind1][5][ self.labels[ind1][1]==ind2 ], (-1,1) )
            cls = np.reshape( self.labels[ind1][0][ self.labels[ind1][1]==ind2 ], (-1,1) )
            label = torch.zeros(5*maxPeaks)
            bh[ bh == 0 ] = self.box_size
            bw[ bw == 0 ] = self.box_size
            #print(r, c, bh, bw)
            try:
                tmp = np.concatenate( (cls, np.maximum(np.minimum(1.0*(c+2)/392.0, 1.0), 0.0), 
                                            np.maximum(np.minimum(1.0*(r+4)/192.0, 1.0), 0.0),
                                           1.0*bw/392.0, 1.0*bh/192.0), axis=1 )
                tmp = torch.from_numpy(tmp)
            except:
                tmp = torch.zeros(1,5)
            
            tmp = tmp.view(-1)
            if r.shape[0] > 0 and tmp.numel() > 5:
                label[0:(5*r.shape[0])] = tmp
            return (timg, label)


class listDataset(Dataset):
    def __init__(self, imgs, labels, shape=None, shuffle=False, predict=False,
                    box_size=7):
       self.imgs = imgs
       self.labels = labels
       self.nSamples  = imgs.shape[0] * imgs.shape[1]  #len(self.lines)
       self.predict = predict
       self.shape = shape
       self.box_size = box_size

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        maxPeaks = 1024
        (n,m,h,w) = self.imgs.shape
#         print("img dims:", (n,m,h,w))
        ind1 = index / m
        ind2 = index % m
#         print("index, m, ind1/ind2", index, m, ind1, ind2 )

        new_h = 192
        new_w = 392
        timg = torch.zeros( ( 1, new_h, new_w) )

        #img = self.imgs[ind1,ind2,:,:] / 15000.0
        img = 1.0 * self.imgs[ind1,ind2,:,:] / np.max(self.imgs[ind1,ind2,:,:])
        img[ img < 0 ] = 0
        timg[:,4:189,2:390] = torch.from_numpy( img )
        timg = timg.view(-1, new_h, new_w )

        if self.predict:
            return timg
        else:
#             print("LABELS:", self.labels)
            r = np.reshape( self.labels[ind1][2][ self.labels[ind1][1]==ind2 ], (-1,1) )
            c = np.reshape( self.labels[ind1][3][ self.labels[ind1][1]==ind2 ], (-1,1) )
            bh = np.reshape( self.labels[ind1][4][ self.labels[ind1][1]==ind2 ], (-1,1) )
            bw = np.reshape( self.labels[ind1][5][ self.labels[ind1][1]==ind2 ], (-1,1) )
            cls = np.reshape( self.labels[ind1][0][ self.labels[ind1][1]==ind2 ], (-1,1) )
#             print("r/c/bh/bw", len(r), len(c), len(bh), len(bw))
#             cls = np.zeros( r.shape )
            label = torch.zeros(5*maxPeaks)
            bh[ bh == 0 ] = self.box_size
            bw[ bw == 0 ] = self.box_size
            #print(r, c, bh, bw)
            try:
                tmp = np.concatenate( (cls, np.maximum(np.minimum(1.0*(c+2)/392.0, 1.0), 0.0), 
                                            np.maximum(np.minimum(1.0*(r+4)/192.0, 1.0), 0.0),
                                           1.0*bw/392.0, 1.0*bh/192.0), axis=1 )
                tmp = torch.from_numpy(tmp)
            except:
                tmp = torch.zeros(1,5)
            
            tmp = tmp.view(-1)
            if r.shape[0] > 0 and tmp.numel() > 5:
                label[0:(5*r.shape[0])] = tmp
#             print("final label [0]:", label[1:5])
#             print("final label [1]:", label[6:10])
#             print("final label [2]:", label[11:15])
#             print("final label [3]:", label[16:20])
            return (timg, label)
