import sys
import os
from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
import cv2

import utils_vis as utils

threshold = 0.9

def maskImage(image_name, category):
    image_path = "../data_large/" + image_name
    mask_path = "../pspnet_prediction/category_mask/" + image_name.replace("jpg","png")

    image = misc.imread(image_path, mode='RGB')
    mask = misc.imread(mask_path)
    category_mask = (mask == category)
    masked_image = utils.apply_mask(image, category_mask)

    question_image = np.hstack((image, masked_image))
    return question_image

def processCategory(cat):
    categories = utils.get_categories()

    name = "{}{}".format(cat,categories[cat])
    question_images_dir = "question_images/" + name
    if not os.path.isdir(question_images_dir):
        os.makedirs(question_images_dir)

    with open("sorted/{}.txt".format(name), 'r') as f:
        counter = 0
        for line in f:
            split = line.split()
            image_name = split[0]
            prob = float(split[1])
            print counter, image_name, prob


            question_name = image_name.replace('/', '-')
            file_path = "{}/{}".format(question_images_dir, question_name)
            if prob > threshold and not os.path.exists(file_path):
                question_image = maskImage(image_name, cat)
                misc.imsave(file_path, question_image)
            
            counter += 1
            if counter == 1000:
                break

cat = int(sys.argv[1])
processCategory(cat)

# for i in xrange(1,151):
#     processCategory(i)
