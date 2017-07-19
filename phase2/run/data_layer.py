import caffe

import os
import numpy as np
import time
from scipy import misc
import random
from data_source import DataSource
from prefetcher import PreFetcher

import utils_run as utils

PRED_DIR = "/data/vision/oliva/scenedataset/scaleplaces/ScalePlaces/phase1/run/predictions/{}/sigmoid/snapshot_iter_{}/"

class DataLayer(caffe.Layer):
    """
    Load (input image, label image) pairs from ADE20K
    """

    def setup(self, bottom, top):
        """
        Setup data layer according to parameters:
        """
        # Always train on ade20k
        project = "ade20k"
        config = utils.get_config(project)
        iter_num = 50000
        config["pspnet_prediction"] = PRED_DIR.format(project, iter_num)

        params = eval(self.param_str)
        self.model = params['model']
        self.batch_size = params['batch_size']
        
        data_source = DataSource(config, random=True)
        image_processor = ImageProcessor(data_source, self.model)
        self.prefetcher = PreFetcher(image_processor, batch_size=self.batch_size, ahead=12)

        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define two tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

    def reshape(self, bottom, top):
        t = time.time()
        self.data, self.label = self.prefetcher.fetch_batch()
        print time.time() - t, self.data.shape, self.label.shape

        top[0].reshape(*self.data.shape)
        top[1].reshape(*self.label.shape)

    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data
        top[1].data[...] = self.label

    def backward(self, top, propagate_down, bottom):
        pass
