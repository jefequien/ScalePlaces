import pdb
import sys, glob, socket, time, os, math, random, argparse
import numpy as np
import scipy, h5py

CAFFE_ROOT = '/data/vision/torralba/segmentation/places/PSPNet/'
sys.path.insert(0, os.path.join(CAFFE_ROOT, 'python'))
import caffe

class PSPNet:
    def __init__(self, DEVICE=0):
        SEED = 3
        caffe.set_mode_gpu()
        caffe.set_device(DEVICE)

        MODEL_INFERENCE = 'pspnet50_ADE20K_473.prototxt'
        WEIGHTS = '/data/vision/torralba/segmentation/places/PSPNet/evaluation/model/pspnet50_ADE20K.caffemodel'
        self.net = caffe.Net(MODEL_INFERENCE, WEIGHTS, caffe.TEST)

        random.seed(SEED)

        fn_log = 'logs/%s_seed%d_gpu%d.log'%(socket.gethostname(), SEED, DEVICE)

    def feed_forward(self, image):
        pass
    def get_network_architecture(self):
        for k,v in self.net.blobs.items():
            print v.data.shape, k

pspnet = PSPNet()
pspnet.get_network_architecture()
