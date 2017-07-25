import os
import sys
import time
import random
import numpy as np
from scipy import misc, ndimage

from image_processor import ImageProcessor
from prefetcher import PreFetcher

CAFFE_ROOT = '/data/vision/torralba/segmentation/places/PSPNet/'
sys.path.insert(0, os.path.join(CAFFE_ROOT, 'python'))
import caffe

import utils_run as utils

class RefineNet:
    def __init__(self, datasource, MODEL, WEIGHTS, DEVICE=0):
        caffe.set_mode_gpu()
        caffe.set_device(DEVICE)

        self.datasource = datasource
        
        model = utils.parse_model(MODEL)
        print MODEL, model
        self.image_processor = ImageProcessor(datasource, model)
        self.prefetcher = PreFetcher(self.image_processor, mode='test', batch_size=None, ahead=4)

        self.test_net = caffe.Net(MODEL, WEIGHTS, caffe.TEST)
        print "Model: ", MODEL
        print "WEIGHTS: ", WEIGHTS

    def process(self, im):
        ap = self.datasource.get_all_prob(im)
        NUM_CLASS,h_ori,w_ori = ap.shape
        slices = self.image_processor.get_slices(ap)

        data = self.image_processor.build_data(im)
        # data = self.prefetcher.fetch_batch()
        out = []
        for s in data:
            o = self.feed_forward(s)
            out.append(o)
        out = np.concatenate(out,axis=0)
        print out.shape

        # Scale back to original size
        _,h,w = out.shape
        out_scaled = ndimage.zoom(out, (1.,1.*h_ori/h,1.*w_ori/w), order=1, prefilter=False)

        output = np.zeros(ap.shape)
        for i in xrange(len(slices)):
            s = slices[i]
            output[s] = out_scaled[i]
        return output

    def feed_forward(self, data):
        self.test_net.blobs['data'].data[...] = data
        self.test_net.forward()
        #conv1 = self.test_net.blobs['conv1'].data[0,:,:,:]
        #print conv1.dtype, conv1.shape, np.mean(conv1)
        #conv1 = conv1.astype('uint') * 255
        #misc.imsave("conv1.png", conv1[0,:,:])

        #W = self.test_net.params['conv1'][0].data[...]
        #print W.shape, np.min(W), np.max(W), np.mean(W)
        #conv1 = self.test_net.blobs['conv1'].data[0,:,:,:]
        #misc.imsave("conv1.png", conv1)
        #conv1 = self.test_net.blobs['conv1'].data[0,:,:,:]
        #misc.imsave("conv1.png", conv1)

        out = self.test_net.blobs['prob'].data[0,:,:,:]
        return np.copy(out)
        
    def test(self, im):
        ap = self.datasource.get_all_prob(im)
        slices = self.image_processor.get_slices(ap)
        s = slices[0]
        print s
        data = self.image_processor.build_data(im)
        data = data[0,:,:,:]
        prob = data[0,:,:]
        canny = data[1,:,:]
        o = self.feed_forward(data)
        o = o[0,:,:]
        print o.dtype, np.min(o), np.max(o), np.mean(o)
        misc.imsave("canny.png", canny)
        misc.imsave("prob.png", prob)
        misc.imsave("{}.png".format(s), o)


