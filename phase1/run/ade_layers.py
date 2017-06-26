import caffe

import os
import numpy as np
from PIL import Image

import random

import utils_run as utils

class AdeSegDataLayer(caffe.Layer):
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
        config = utils.get_data_config(project)
        self.data_dir = os.path.join(config["pspnet_prediction"], "all_prob")
        self.label_dir = config["ground_truth"]

        # config
        params = eval(self.param_str)
        self.split = params['split']
        txt_imlist = "{}.txt".format(self.split)
        self.im_list = [line.rstrip() for line in open(txt_imlist, 'r')]
        self.idx = 0

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
        if 'train' not in self.split:
            self.random = False

        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            self.idx = random.randint(0, len(self.indices)-1)


    def reshape(self, bottom, top):
        # load image + label image pair
        self.data = self.load_image(self.im_list[self.idx])
        self.label = self.load_label(self.im_list[self.idx])
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


    def load_image(self, im):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        img = Image.open(os.path.join(self.data_dir, im))
        in_ = np.array(img, dtype=np.float32)
        if (in_.ndim == 2):
            in_ = np.repeat(in_[:,:,None], 3, axis = 2)
        in_ = in_[:,:,::-1]
        in_ -= self.mean
        in_ = in_.transpose((2,0,1))
        return in_


    def load_label(self, im):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        img = Image.open(os.path.join(self.label_dir, im.replace(".jpg",".png")))
        label = np.array(img, dtype=np.uint8)
        label = label[np.newaxis, ...]
        return label
