import caffe

import os
import numpy as np
from scipy import misc

import random

import utils_run as utils

class DataLayer(caffe.Layer):
    """
    Load (input image, label image) pairs from PASCAL VOC
    one-at-a-time while reshaping the net to preserve dimensions.
    Use this to feed data to a fully convolutional network.
    """

    def setup(self, bottom, top):
        """
        Setup data layer according to parameters:
        - ade_dir: path to ADE dir
        - split: train / val / test
        - mean: tuple of mean values to subtract
        - randomize: load in random order (default: True)
        - seed: seed for randomization (default: None / current time)
        
        """
        project = "ade20k"
        CONFIG = utils.get_data_config(project)
        self.image_dir = CONFIG["images"]
        self.label_dir = CONFIG["ground_truth"]
        self.all_prob_dir = os.path.join(CONFIG["pspnet_prediction"], "all_prob")
        self.im_list = [line.rstrip() for line in open(CONFIG["im_list"], 'r')]

        # Params
        self.idx = 0
        params = eval(self.param_str)
        self.mean = np.array(params['mean'])
        self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)

        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define two tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # make eval deterministic
        self.random = False

        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            self.idx = random.randint(0, len(self.im_list)-1)


    def reshape(self, bottom, top):
        # load image + label image pair
        data = self.load_image(self.im_list[self.idx])
        label = self.load_label(self.im_list[self.idx])
        print data.shape, label.shape
        self.data, self.label = self.transform(data, label)
        print self.data.shape, self.label.shape

        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(1, *self.data.shape)
        top[1].reshape(1, *self.label.shape)


    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data
        top[1].data[...] = self.label

        # pick next input
        if self.random:
            self.idx = random.randint(0, len(self.im_list)-1)
        else:
            self.idx += 1
            if self.idx == len(self.im_list):
                self.idx = 0


    def backward(self, top, propagate_down, bottom):
        pass

    def transform(self, data, label):
        # Resize
        base_size = 512
        data = utils.resize(data, base_size)
        label = utils.resize(label, base_size, interp='nearest')

        # Random crop
        crop_size = 473
        h,w,n = data.shape
        dx = random.randint(0,w-crop_size)
        dy = random.randint(0,h-crop_size)
        data = data[dy:473+dy,dx:473+dx]
        label = label[dy:473+dy,dx:473+dx]

        # Make label
        K = 150
        new_label = np.zeros((K,crop_size,crop_size))
        for i in xrange(K):
            c = i+1
            new_label[i] = label == c
       
        data = data.transpose((2,0,1))
        label = label[np.newaxis, ...]
        return data, label

    def load_image(self, im):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - subtract mean
        """
        img = misc.imread(os.path.join(self.image_dir, im))
        in_ = np.array(img, dtype=np.float32)

        if (in_.ndim == 2):
            in_ = np.repeat(in_[:,:,None], 3, axis = 2)
        in_ -= self.mean
        return in_

    def load_label(self, im):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        img = misc.imread(os.path.join(self.label_dir, im.replace(".jpg",".png")))
        label = np.array(img, dtype=np.uint8)
        return label
