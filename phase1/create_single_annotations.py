import sys
import os
from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
import cv2

import utils

threshold = 0.9

def maskImage(image_name, category):
    image_path = "../data_large/" + image_name
    mask_path = "../pspnet_prediction/category_mask/" + image_name.replace("jpg","png")

    image = misc.imread(image_path, mode='RGB')
    mask = misc.imread(mask_path)
    category_mask = (mask == category)
    masked_image = utils.apply_mask(image, category_mask)
    return masked_image

def processCategory(cat):
    categories = utils.get_categories()

    name = "{}{}".format(cat,categories[cat])
    output_dir = "single_annotations/" + name
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    with open("sorted/{}.txt".format(name), 'r') as f:
        percentiles = [1.,.9,.8,.7,.6,.5,.4,.3,.2,.1]
        for p in percentiles:
            percentile_dir = "{}/{}".format(output_dir, int(p*100))
            if not os.path.isdir(percentile_dir):
                os.makedirs(percentile_dir)

            counter = 0
            for line in f:
                split = line.split()
                image_name = split[0]
                prob = float(split[1])

                if p >= prob and prob > p - 0.1:
                    file_name = image_name.replace('/', '-')
                    file_path = "{}/{}".format(percentile_dir, file_name)

                    if not os.path.exists(file_path):
                        masked_image = maskImage(image_name, cat)
                        misc.imsave(file_path, masked_image)
                    
                    print cat, p, image_name, prob

                    counter += 1
                    if counter == 100:
                        break
                if prob < p - 0.1:
                    break

# cat = int(sys.argv[1])
# processCategory(cat)

for i in xrange(1,151):
    processCategory(i)
