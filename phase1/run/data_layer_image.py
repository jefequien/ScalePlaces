import caffe

import os
import numpy as np
from scipy import misc

import random

import utils_run as utils
import utils_pspnet

class DataLayer(caffe.Layer):
    """
    Load (input image, label image) pairs from PASCAL VOC
    one-at-a-time while reshaping the net to preserve dimensions.
    Use this to feed data to a fully convolutional network.
    """

    def setup(self, bottom, top):
        """
        Setup data layer according to parameters:
        
        """
        project = "ade20k"
        CONFIG = utils.get_config(project)
        self.im_list = utils.open_im_list(project)
        self.image_dir = CONFIG["images"]
        self.gt_dir = CONFIG["ground_truth"]

        # Params
        self.idx = 0
        self.random = False
        self.seed = 1337

        params = eval(self.param_str)
        self.loss_type = params['loss_type']

        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define two tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            self.idx = random.randint(0, len(self.im_list)-1)


    def reshape(self, bottom, top):
        # load image + label image pair
        im = self.im_list[self.idx]
        img = self.load_image(im)
        gt = self.load_ground_truth(im)

        self.data, self.label = self.transform(img, gt)

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

    def transform(self, img, gt):
        img = utils_pspnet.scale(img)
        gt = utils_pspnet.scale(gt, interp='nearest')

        # Random crop
        crop_size = 473
        h,w,n = img.shape

        sh = 0
        sw = 0
        if h > crop_size:
            sh = random.randint(0,h-crop_size)
        if w > crop_size:
            sw = random.randint(0,w-crop_size)
        eh = min(h,sh + crop_size)
        ew = min(w,sw + crop_size)

        box = (sh,eh,sw,ew)
        img = utils_pspnet.crop_image(img, box)
        gt = utils_pspnet.crop_gt(gt, box)

        # Make label from gt
        label = None
        if self.loss_type == "softmax":
            label = gt
            # Make ignored category 150
            label[label==0] = 151
            label -= 1

        elif self.loss_type == "sigmoid":
            K = 150
            label = np.zeros((K,crop_size,crop_size))
            # Ignore category 2
            label.fill(2)

            for i in xrange(K):
                c = i+1
                mask = gt == c
                if np.sum(mask) > 0:
                    label[i] = mask
        else:
            raise
       
        data = data[:,:,(2,1,0)]
        data = data.transpose((2,0,1))
        return data, label

    def load_image(self, im):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - subtract mean
        """
        img = misc.imread(os.path.join(self.image_dir, im))
        in_ = utils_pspnet.preprocess(img)
        return in_

    def load_ground_truth(self, im):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        gt_path = os.path.join(self.gt_dir, im.replace(".jpg",".png"))
        gt = misc.imread(gt_path)
        gt = np.array(gt, dtype=np.uint8)
        return gt
