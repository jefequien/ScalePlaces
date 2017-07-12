import os
import sys
import time
import random
import numpy as np

from params import WEIGHTS, MODEL
from utils_pspnet import *

CAFFE_ROOT = '/data/vision/torralba/segmentation/places/PSPNet/'
sys.path.insert(0, os.path.join(CAFFE_ROOT, 'python'))
import caffe

class PSPNet:
    def __init__(self, DEVICE=0):
        caffe.set_mode_gpu()
        caffe.set_device(DEVICE)

        SEED = 3
        random.seed(SEED)

        self.test_net = caffe.Net(MODEL, WEIGHTS, caffe.TEST)
        print "Model: ", MODEL
        print "WEIGHTS: ", WEIGHTS

    def sliding_window(self, image):
        image = preprocess(image)
        h_ori,w_ori,_ = image.shape

        # Scale image to fixed size
        image_scaled = scale_image(image)
        # Split image into crops with sliding window
        crops = split_crops(image_scaled)

        n = crops.shape[0]
        crop_probs = None
        for i in xrange(n):
            crop = crops[i]
            crop_prob = self.feed_forward(crop)

            K,h,w = crop_prob.shape
            if crop_probs is None:
                crop_probs = np.zeros((n,K,h,w))
            crop_probs[i] = crop_prob

        # Reconstruct probability crops together
        probs = assemble_probs(image_scaled,crop_probs)
        # Scale back to original size
        probs = unscale(probs,h_ori,w_ori)
        return probs

    def feed_forward(self, data):
        '''
        Input must be 473x473x3 in RGB
        Output is 150x473x473
        '''
        assert data.shape == (473,473,3)
        # RGB => BGR
        data = data[:,:,(2,1,0)]
        data = data.transpose((2,0,1))
        data = data[np.newaxis,:,:,:]

        self.test_net.blobs['data'].data[...] = data
        self.test_net.forward()
        out = self.test_net.blobs['prob'].data[0,:,:,:]
        return np.copy(out)
        

    def print_network_architecture(self):
        for k,v in self.test_net.blobs.items():
            print v.data.shape, k
