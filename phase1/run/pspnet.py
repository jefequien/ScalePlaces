import os
import sys
import time
import random
import socket
import numpy as np

import utils_run as utils
import pspnet_utils

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

        # MODEL_INFERENCE = 'models/train_pspnet_modified.prototxt'
        MODEL_INFERENCE = 'models/pspnet50_ADE20K_473.prototxt'

        self.net = caffe.Net(MODEL_INFERENCE, WEIGHTS, caffe.TEST)

        self.log = 'logs/%s_seed%d_gpu%d.log'%(socket.gethostname(), SEED, DEVICE)

    def fine_tune(self):
        solver = caffe.get_solver('models/solver_pspnet_modified.prototxt')
        solver.net.copy_from(WEIGHTS)


    def old_process(self, image):
        import time
        import numpy as np
        import itertools
        from scipy import misc, ndimage
        DATA_MEAN = np.array([[[123.68, 116.779, 103.939]]])
        INPUT_SIZE = 473
        NUM_CLASS = 150

        image = pspnet_utils.preprocess(image)
        
        # Resize consistent size
        h_ori,w_ori,n = image.shape

        image = pspnet_utils.scale(image)

        h,w,_ = image.shape
        crops = pspnet_utils.split_crops(image)

        probs = np.zeros((NUM_CLASS, h, w), dtype=np.float32)
        cnts = np.zeros((1,h,w))

        n = len(crops)
        outs = []
        for i in xrange(n):
            data = crops[i]

            out = self.feed_forward(data)
            outs.append(out)


        crop_boxes = pspnet_utils.get_crop_boxes(h,w)
        for i in xrange(n):
            sh,eh,sw,ew = crop_boxes[i]


            cnts[0,sh:eh,sw:ew] += 1
            probs[:,sh:eh,sw:ew] += out[:,0:eh-sh,0:ew-sw]

        probs /= cnts

        # Resize back
        probs = pspnet_utils.unscale(probs, h_ori, w_ori)
        assert probs.shape == (NUM_CLASS,h_ori,w_ori)
        return probs

    def process(self, image):
        image = pspnet_utils.preprocess(image)
        h_ori,w_ori,_ = image.shape

        image_scaled = pspnet_utils.scale(image)
        crops = pspnet_utils.split_crops(image_scaled)

        crop_probs = []
        for crop in crops:
            crop_prob = self.feed_forward(crop)
            crop_probs.append(crop_prob)

        probs = pspnet_utils.assemble_probs(image_scaled,crop_probs)
        probs = pspnet_utils.unscale(probs,h_ori,w_ori)

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

        self.net.blobs['data'].data[...] = data
        self.net.forward()
        out = self.net.blobs['prob'].data[0,:,:,:]
        return out
        

    def print_network_architecture(self):
        for k,v in self.net.blobs.items():
            print v.data.shape, k








