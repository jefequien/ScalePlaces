import sys
import os
from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
import cv2

import utils

threshold = 0.9999

temp_image = "/Users/hujh/Documents/UROP_Torralba/ADE_20K/images/ADE_train_00000037.jpg"
temp_mask = "/Users/hujh/Documents/UROP_Torralba/ADE_20K/annotations/ADE_train_00000037.png"

def maskImage(image_path, category):
    image_path = temp_image
    mask_path = temp_mask

    image = misc.imread(image_path)
    mask = misc.imread(mask_path)
    category_mask = (mask == category)
    category_mask = cv2.cvtColor(category_mask.astype('uint8')*255, cv2.COLOR_GRAY2RGB)

    question_image = np.vstack((image, category_mask))
    return question_image


cat = int(sys.argv[1])
categories = utils.get_categories()

name = "{}{}".format(cat,categories[cat])
question_images_dir = "question_images/" + name
if not os.path.isdir(question_images_dir):
    os.makedirs(question_images_dir)

with open("sorted/{}.txt".format(name), 'r') as f:
    for line in f:
        split = line.split()
        image_path = split[0]
        prob = float(split[1])
        print image_path, prob

        if prob > threshold:
            question_image = maskImage(image_path, cat)
            question_name = image_path.replace('/', '#')
            misc.imsave("{}/{}".format(question_images_dir, question_name), question_image)