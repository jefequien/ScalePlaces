import os
import time
import random
import h5py
from scipy import misc

import utils_run as utils
from canny import *

class DataSource:
    def __init__(self, config, random=True):
        self.image_dir = config["images"]
        self.all_prob_dir = os.path.join(config["pspnet_prediction"], "all_prob")
        # self.canny_dir = config["canny"]
        self.ground_truth_dir = config["ground_truth"]

        im_list_txt = config["im_list"]
        self.im_list = utils.open_im_list(im_list_txt)
        self.idx = 0
        self.random = random

    def next_idx(self):
        idx = self.idx
        if not random:
            self.idx += 1
            if self.idx == len(self.im_list):
                self.idx = 0
        else:
            self.idx = random.randint(0,len(self.im_list))
        return idx

    def get_image(self, idx):
        im = self.im_list[idx]
        img_path = os.path.join(self.image_dir, im)
        return misc.imread(img_path)
    def get_ground_truth(self, idx):
        im = self.im_list[idx]
        gt_path = os.path.join(self.ground_truth_dir, im.replace('.jpg', '.png'))
        return misc.imread(gt_path)

    def get_all_prob(self, idx):
        im = self.im_list[idx]
        ap_path = os.path.join(self.all_prob_dir, im.replace('.jpg', '.h5'))
        with h5py.File(ap_path, 'r') as f:
            output = f['allprob'][:]
            return output

    def get_canny(self, idx):
        im = self.im_list[idx]
        canny_path = os.path.join(self.canny_dir, im.replace('.jpg', '.png'))
        if not os.path.exists(canny_path):
            print "Missing canny"
            if not os.path.exists(os.path.dirname(canny_path)):
                os.makedirs(os.path.dirname(canny_path))

            t = time.time()
            img = self.get_image(idx)
            edges = canny(img)
            misc.imsave(canny_path, edges)
            print time.time() - t
        return misc.imread(canny_path)

