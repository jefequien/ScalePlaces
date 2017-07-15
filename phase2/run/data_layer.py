import caffe

import os
import numpy as np
import time
from scipy import misc
import random
from data_source import DataSource

import utils_run as utils
import utils_pspnet

class DataLayer(caffe.Layer):
    """
    Load (input image, label image) pairs from ADE20K
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

        params = eval(self.param_str)
        self.loss_type = params['loss_type']

        random = True
        data_source = DataSource(config, random=random)
        self.batch_builder = BatchBuilder(data_source, batch_size=4)
        # self.data_loader = PreFetcher(batch_builder)

        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define two tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

    def reshape(self, bottom, top):
        self.data, self.label = self.batch_builder.build_batch()
        # self.data, self.label = self.prefetch.fetch_batch()

        top[0].reshape(self.data.shape)
        top[1].reshape(self.label.shape)

    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data
        top[1].data[...] = self.label

    def backward(self, top, propagate_down, bottom):
        pass