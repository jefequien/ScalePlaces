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

        # Only used for eval
        MODEL_INFERENCE = 'models/pspnet50_ADE20K_473.prototxt'
        self.test_net = caffe.Net(MODEL_INFERENCE, WEIGHTS, caffe.TEST)


        self.log = 'logs/%s_seed%d_gpu%d.log'%(socket.gethostname(), SEED, DEVICE)

    def fine_tune(self):
        solver = caffe.get_solver('models/solver_pspnet_with_data_layer.prototxt')
        solver.net.copy_from(WEIGHTS)
        solver.solve()

    def sliding_window(self, image):
        image = utils_pspnet.preprocess(image)
        h_ori,w_ori,_ = image.shape

        image_scaled = utils_pspnet.scale(image)
        crops = utils_pspnet.split_crops(image_scaled)

        n,h,w,_ = crops.shape
        K = 150
        crop_probs = np.zeros((n,K,h,w))
        for i in xrange(n):
            crop = crops[i]
            crop_probs[i] = self.feed_forward(crop)

        probs = utils_pspnet.assemble_probs(image_scaled,crop_probs)
        probs = utils_pspnet.unscale(probs,h_ori,w_ori)

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








