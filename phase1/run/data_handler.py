import os
import random
import scipy

import sys
sys.path.append(os.path.abspath('../utils'))
import utils

class DataHandler:
    def __init__(self, project, im_list):
        self.project = project
        self.im_list = [line.rstrip().split()[0] for line in open(imlist, 'r')]
        random.shuffle(self.im_list)

        config = utils.get_data_config(project)
        self.root_images = config["images"]
        self.root_cm = os.path.join(config["pspnet_prediction"], "category_mask")
        self.root_pm = os.path.join(config["pspnet_prediction"], "prob_mask")

    def next_image(self):
        im = self.im_list.pop(0)
        image = scipy.misc.imread(os.path.join(self.root_images, im))
        return image

    def save(self, im, ):
        pass