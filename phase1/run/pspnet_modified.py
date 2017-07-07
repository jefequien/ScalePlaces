import os
import sys
import time
import random
import socket
import numpy as np

import utils_run as utils
import utils_pspnet

CAFFE_ROOT = '/data/vision/torralba/segmentation/places/PSPNet/'
sys.path.insert(0, os.path.join(CAFFE_ROOT, 'python'))
import caffe

WEIGHTS = '/data/vision/torralba/segmentation/places/PSPNet/evaluation/model/pspnet50_ADE20K.caffemodel'

class PSPNet:
    def __init__(self, DEVICE=0):
        caffe.set_mode_gpu()
        caffe.set_device(DEVICE)

        SEED = 3
        random.seed(SEED)

        self.log = 'logs/%s_seed%d_gpu%d.log'%(socket.gethostname(), SEED, DEVICE)

    def fine_tune(self):
        solver = caffe.get_solver('models/solver_pspnet_with_data_layer.prototxt')
        solver.net.copy_from(WEIGHTS)
        solver.solve()

    def print_network_architecture(self):
        for k,v in self.test_net.blobs.items():
            print v.data.shape, k








