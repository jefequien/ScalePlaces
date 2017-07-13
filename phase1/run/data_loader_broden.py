import caffe

import os
import numpy as np
import time
from scipy import misc

import random

import utils_run as utils
import utils_pspnet

ROOT = "/data/vision/torralba/deepscene/david_data/bulk/uniseg4_384/"

class BrodenDataLoader():
    """
    Prepares Broden dataset for loading into caffe.
        Resizes images to 473x473x3 RGB
        Creates the sigmoid label.
    """
    def setup(self, bottom, top):
        """
        Setup data layer according to parameters:
        
        """
        project = "broden"
        CONFIG = utils.get_config(project)
        self.im_list = utils.open_im_list(project)
        self.image_dir = CONFIG["images"]
        self.gt_dir = CONFIG["ground_truth"]

        # Params
        self.idx = 0
        self.random = True
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

        self.t = time.time()

    def reshape(self, bottom, top):
        #print "Backprob", time.time() - self.t
        #self.t = time.time()
        
        self.next_idx()
        im = self.im_list[self.idx]
        print im
        
        # Load img and gt
        img = self.load_image(im)
        gt = self.load_ground_truth(im)
        
        self.data, self.label = self.transform(img, gt)

        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(1, *self.data.shape)
        top[1].reshape(1, *self.label.shape)
        
        #print "Overhead", time.time() - self.t
        #self.t = time.time()

    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data
        top[1].data[...] = self.label

    def next_idx(self):
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
        img = utils_pspnet.scale_image(img)
        gt = utils_pspnet.scale_ground_truth(gt)
        
        # Random crop
        box = utils_pspnet.random_crop(img)
        img = utils_pspnet.crop_image(img, box)
        gt = utils_pspnet.crop_ground_truth(gt, box)

        # Setup data
        data = img[:,:,(2,1,0)]
        data = data.transpose((2,0,1))

        # Setup label
        label = None
        if self.loss_type == "softmax":
            label = gt - 1
        elif self.loss_type == "sigmoid":
            # One hot encode 2D array
            NUM_CLASS = 150
            label = (np.arange(NUM_CLASS) == gt[:,:,None] - 1)
            label = label.transpose((2,0,1))
            # Ignore blank slices
            #for i in xrange(NUM_CLASS):
            #    if np.sum(label[i]) == 0:
            #        label[i] = 0
        elif self.loss_type == "specific":
            c = 1
            label = gt == c
        else:
            print "Wrong loss type"
            raise
        return data, label

    def load_image(self, im):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - subtract mean
        - make into 3 channel image
        """
        img = misc.imread(os.path.join(self.image_dir, im))
        in_ = utils_pspnet.preprocess(img)
        return in_

    def load_ground_truth(self, im):
        """
        Load ground_truth image as height x width integer array of label indices.
        """
        gt_path = os.path.join(self.gt_dir, im.replace(".jpg",".png"))
        gt = misc.imread(gt_path)
        gt = np.array(gt, dtype=np.int32)
        return gt
