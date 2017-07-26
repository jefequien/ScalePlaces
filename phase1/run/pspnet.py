import os
import sys
import argparse
import time
import random
import numpy as np
from scipy import misc

from utils_pspnet import *
import utils_run as utils

CAFFE_ROOT = '/data/vision/torralba/segmentation/places/PSPNet/'
sys.path.insert(0, os.path.join(CAFFE_ROOT, 'python'))
import caffe

class PSPNet:
    def __init__(self, MODEL, WEIGHTS, DEVICE=0):
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
        blobs = self.test_net.blobs
        for name in blobs:
            data = blobs[name].data[0,:,:,:]
            print name, data.shape, np.min(data), np.max(data), np.mean(data)
        
        out = self.test_net.blobs['prob'].data[0,:,:,:]
        return np.copy(out)
        
    def print_caffe_model(self):
        params = self.test_net.params
        for name in params:
            print name, params[name]


    def print_network_architecture(self):
        for k,v in self.test_net.blobs.items():
            print v.data.shape, k

def add_color(img):
    h,w = img.shape
    img_color = np.zeros((h,w,3))
    for i in xrange(1,151):
        img_color[img == i] = utils.to_color(i)
    return img_color
    

if __name__ == "__main__":
    WEIGHTS = '/data/vision/torralba/segmentation/places/PSPNet/evaluation/model/pspnet50_ADE20K.caffemodel'
    MODEL = 'models/test_pspnet_softmax.prototxt'
    pspnet = PSPNet(MODEL, WEIGHTS)
    #pspnet.print_caffe_model()

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='', 
                        required=True, help='Path the input image')
    parser.add_argument('--output_path', type=str, default='',
                        required=True, help='Path to output')
    args = parser.parse_args()
    img = misc.imread(args.input_path)
    img = misc.imresize(img, (473,473))
    img = preprocess(img)
    probs = pspnet.feed_forward(img)
    pred_mask = np.argmax(probs, axis=0) + 1

    color_mask = add_color(pred_mask)
    misc.imsave(args.output_path, color_mask)


