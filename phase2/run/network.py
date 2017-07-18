import os
import sys
import time
import random
import numpy as np

from image_processor import ImageProcessor
from utils_pspnet import *

CAFFE_ROOT = '/data/vision/torralba/segmentation/places/PSPNet/'
sys.path.insert(0, os.path.join(CAFFE_ROOT, 'python'))
import caffe

class Network:
    def __init__(self, datasource, MODEL, WEIGHTS, DEVICE=0):
        caffe.set_mode_gpu()
        caffe.set_device(DEVICE)

        self.datasource = datasource
        self.image_processor = ImageProcessor(datasource)

        self.test_net = caffe.Net(MODEL, WEIGHTS, caffe.TEST)
        print "Model: ", MODEL
        print "WEIGHTS: ", WEIGHTS

    def process(self, idx):
        ap = self.datasource.get_all_prob(idx)
        NUM_CLASS,h_ori,w_ori = ap.shape
        slices = self.datasource.get_slices(ap)

        data = self.image_processor.build_data(idx)

        self.test_net.blobs['data'].data[...] = data
        self.test_net.forward()
        out = self.test_net.blobs['prob'].data[:,:,:,:]

        _,h,w = out.shape
        out_scaled = ndimage.zoom(out, (1.,1.*h_ori/h,1.*w_ori/w), order=1, prefilter=False)

        output = np.zeros(ap.shape)
        for i in xrange(len(slices)):
            s = slices[i]
            output[s] = out_scaled[i]
        return output
