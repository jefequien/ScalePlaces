import os
import json
import h5py
from scipy import misc
import matplotlib.pyplot as plt
import cv2
import numpy as np
import colorsys
import random

RED = [255,0,0]
GREEN = [0,192,0]
BLUE = [0,0,255]

COLORS = [RED,GREEN,BLUE]

PATH = os.path.dirname(__file__)

categories = get_categories()

def get_config(project):
    with open(os.path.join(PATH, "../../../LabelMe/data_config.json"), 'r') as f:
        data_config = json.load(f)
        config = data_config[project]
        return config

# Can also be project
def open_im_list(txt_im_list):
    if ".txt" not in txt_im_list:
        project = txt_im_list
        CONFIG = get_config(project)
        txt_im_list = CONFIG["im_list"]

    im_list = [line.rstrip() for line in open(txt_im_list, 'r')]
    return im_list

def get_categories():
    categories = {}
    with open(os.path.join(PATH, "objectInfo150.txt"), 'r') as f:
        for line in f.readlines():
            split = line.split()
            cat = split[0]
            if cat.isdigit():
                categories[int(cat)] = split[4].replace(',','')
        return categories

def to_color(category):
    v = (category-1)*(137.5/360)
    return colorsys.hsv_to_rgb(v,1,1)


def apply_mask(image, mask):
    masked_image = np.copy(image)

    masked_image[mask] = np.maximum(masked_image[mask], random.choice(COLORS))

    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(masked_image, contours, -1, (0, 0, 0), 2)
    return masked_image

def get_file(im, config, ftype="im"):
    file_path = get_file_path(im, config, ftype=ftype)
    if ftype == "im":
        return misc.imread(file_path)
    elif ftype == "gt":
        return misc.imread(file_path)
    elif ftype == "cm":
        return misc.imread(file_path)
    elif ftype == "ap":
        with h5py.File(file_path, 'r') as f:
            output = f['allprob'][:]
            return output
    elif ftype == "pm":
        return misc.imread(file_path)
    else:
        print "File type not found."
        raise Exception

def get_file_path(im, config, ftype="im"):
    if ftype == "im":
        root = config["images"]
        return os.path.join(root, im)
    elif ftype == "gt":
        root = config["ground_truth"]
        return os.path.join(root, im.replace(".jpg",".png"))
    elif ftype == "cm":
        root = os.path.join(config["pspnet_prediction"], "category_mask")
        return os.path.join(root, im.replace(".jpg",".png"))
    elif ftype == "ap":
        root = os.path.join(config["pspnet_prediction"], "all_prob")
        fname = os.path.join(root, im.replace(".jpg", ".h5"))
        return fname
    elif ftype == "pm":
        root = os.path.join(config["pspnet_prediction"], "prob_mask")
        fname = os.path.join(root, im)
        return fname
    else:
        print "File type not found."
        raise Exception


if __name__=="__main__":
    image_path = "/Users/hujh/Documents/UROP_Torralba/ADE_20K/images/ADE_train_00000037.jpg"
    mask_path = "/Users/hujh/Documents/UROP_Torralba/ADE_20K/annotations/ADE_train_00000037.png"
    category = 1
    image = misc.imread(image_path)
    mask = misc.imread(mask_path)

    category_mask = (mask == category)
    masked_image = apply_mask(image, category_mask)

    plt.imshow(masked_image)
    plt.show()
