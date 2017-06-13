import sys
from scipy import misc

import utils

threshold = 0.9

temp_image = "/Users/hujh/Documents/UROP_Torralba/ADE_20K/images/ADE_train_00000037.jpg"
temp_mask = "/Users/hujh/Documents/UROP_Torralba/ADE_20K/annotations/ADE_train_00000037.png"

def processImage(image_path, category):
    print image_path, category

    image_path = temp_image
    mask_path = temp_mask

    image = misc.imread(image_path)
    mask = misc.imread(mask_path)

    category_mask = mask

cat = int(sys.argv[1])
categories = utils.get_categories()

with open("{}{}.txt".format(cat,categories[cat]), 'r') as f:
    for line in f:
        split = line.split()
        image_path = split[0]
        prob = float(split[1])
        print image_path, prob
        if prob > threshold:
            annotated_image = processImage(image_path, cat)
            break

