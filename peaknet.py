import os
import os.path as osp
import time
import torch as t
import sys
import numpy as np
import torch
from darknet import Darknet
import peaknet_train
from peaknet_validate import validate_batch
from peaknet_predict import predict_batch
from tensorboardX import SummaryWriter


class Peaknet():

    def __init__(self):
        self.model = None
        self.optimizer = None
        self.writer = None

    def set_writer(self, project_name=None, parameters={}):
        if project_name == None:
            self.writer = SummaryWriter()
        else:
            self.writer = SummaryWriter( project_name, purge_step=0 )

    def loadCfg( self, cfgFile ):
        self.model = Darknet( cfgFile )

    def loadWeights( self, cfgFile, weightFile ):
        self.model = Darknet( cfgFile )
        self.model.load_weights( weightFile )

    def loadDNWeights( self, weight_path ):
        self.model.load_weights(weight_path )
        
    def snapshot( self, imgs, labels, path, tag="snapshot" ):
        seen = self.model.seen
        np.save(os.path.join(path, "{}_imgs_{}".format(tag,
                str(seen).zfill(9))), imgs)
        np.save(os.path.join(path, "{}_labels_{}".format(tag,
                str(seen).zfill(9))), labels)
        torch.save(self.model, os.path.join(path,
                "{}_model_{}".format(tag, str(seen).zfill(9))))

    def init_model( self ):
        peaknet_train.init_model( self.model )

    def train( self, imgs, labels, box_size = 7, mini_batch_size=1,
            use_cuda=True, writer=None, verbose=False ):
        self.model.delta = self.model.seen
        peaknet_train.train_batch( self.model, imgs, labels,
                mini_batch_size=mini_batch_size, box_size=box_size,
                use_cuda=use_cuda, writer=self.writer, verbose=verbose )
        self.model.delta -= self.model.seen

    def model( self ):
        return self.model

    def getGrad( self ):
        grad = {}
        model_dict = dict( self.model.named_parameters() )
        for key, val in model_dict.items():
            grad[key] = val.grad.cpu()
        return grad

    def predict( self, imgs, box_size = 7, batch_size=1, conf_thresh=0.15,
            nms_thresh=0.45, use_cuda=True ):
        results = predict_batch( self.model, imgs, batch_size=batch_size,
                conf_thresh=conf_thresh, nms_thresh=nms_thresh,
                box_size=box_size, use_cuda=use_cuda)
        return results

    def validate( self, imgs, labels, box_size = 7, mini_batch_size=32,
            use_cuda=True, writer=None, verbose=False ):
        recall = validate_batch( self.model, imgs, labels, json_file=None,
            mini_batch_size=mini_batch_size, box_size=box_size,
            use_cuda=use_cuda, writer=writer, verbose=verbose )
        return recall
    
    def validate_psana( self, json_file, box_size = 7, mini_batch_size=32,
            use_cuda=True, writer=None, verbose=False ):
        recall = validate_batch( self.model, imgs=None, labels=None,
                json_file=json_file, mini_batch_size=mini_batch_size, 
                box_size=box_size, use_cuda=use_cuda, writer=writer,
                verbose=verbose)
        return recall

    def updateModel( self, model, check=False ):
        if check:
            model_dict = dict( self.model.named_parameters() )
            model_dict2 = dict( model.named_parameters() )
            for key, value in model_dict.items():
                if model_dict[key] != model_dict2[key]:
                    print("{} updated".format(key))
                    pass
                else:
                    print("{} didn't get updated".format(key))
        delta = model.delta
        seen = self.model.seen + delta
        self.model = model
        self.model.seen = seen

    def updateGrad( self, grads, delta=0, useGPU=False):
        peaknet_train.updateGrad( self.model, grads, delta=delta,
                useGPU=useGPU )

    def set_optimizer( self, adagrad=False, lr=0.001 ):
        self.optimizer = peaknet_train.optimizer( self.model, adagrad=adagrad,
                lr=lr )

    def optimize( self ):
        peaknet_train.optimize( self.model, self.optimizer )
