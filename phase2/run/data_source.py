import os
import time
import random
import h5py
import numpy as np
from scipy import misc, special

import utils_run as utils
from canny import *

class DataSource:
    def __init__(self, config, random=True):
        self.image_dir = config["images"]
        self.all_prob_dir = os.path.join(config["pspnet_prediction"], "all_prob")
        self.ground_truth_dir = config["ground_truth"]
        self.canny_dir = os.path.join(config["workspace"], "canny")

        im_list_txt = config["im_list"]
        self.im_list = utils.open_im_list(im_list_txt)
        
        self.random = random
        if not self.random:
            self.idx = -1

    def next_im(self):
        if self.random:
            idx = random.randint(0,len(self.im_list)-1)
            return self.im_list[idx]
        else:
            self.idx += 1
            if self.idx == len(self.im_list):
                self.idx = 0
            return self.im_list[self.idx]

    def get_image(self, im):
        img_path = os.path.join(self.image_dir, im)
        img = misc.imread(img_path)
        
        if img.ndim != 3:
            img = np.stack((img,img,img), axis=2)
        img = img.transpose((2,0,1))
        return img

    def get_ground_truth(self, im):
        gt_path = os.path.join(self.ground_truth_dir, im.replace('.jpg', '.png'))
        return misc.imread(gt_path)

    def get_all_prob(self, im):
        ap_path = os.path.join(self.all_prob_dir, im.replace('.jpg', '.h5'))
        with h5py.File(ap_path, 'r') as f:
            output = f['allprob'][:]
            #output = (output*255).astype('uint8')
            output = special.logit(output)
            output = np.clip(output, -6,6)
            return output

    def get_canny(self, im):
        canny_path = os.path.join(self.canny_dir, im.replace('.jpg', '.png'))
        if not os.path.exists(canny_path):
            if not os.path.exists(os.path.dirname(canny_path)):
                os.makedirs(os.path.dirname(canny_path))
            print im

            t = time.time()
            img = self.get_image(im)
            img = img.transpose((1,2,0))
            edges = make_canny(img)
            misc.imsave(canny_path, edges)
            
        canny = misc.imread(canny_path)
        canny = canny.astype('float32')
        canny /= 255.
        canny = canny[np.newaxis,:,:]
        return canny

