import os
import json
import random

PATH = os.path.dirname(__file__)

def get_categories():
    categories = {}
    with open(os.path.join(PATH, "objectInfo150.txt"), 'r') as f:
        for line in f.readlines():
            split = line.split()
            cat = split[0]
            if cat.isdigit():
                categories[int(cat)] = split[4].replace(',','')
        return categories

categories = get_categories()

def get_config(project):
    with open(os.path.join(PATH, "../../../LabelMe/data_config.json"), 'r') as f:
        data_config = json.load(f)
        config = data_config[project]
        return config

# Can also be project
def open_im_list(im_list_txt, r=False):
    if ".txt" not in im_list_txt:
        project = im_list_txt
        CONFIG = get_config(project)
        im_list_txt = CONFIG["im_list"]

    im_list = [line.rstrip() for line in open(im_list_txt, 'r')]
    if r:
        seed = 3
        random.seed(seed)
        random.shuffle(im_list)
    return im_list

def to_color(category):
    v = (category-1)*(137.5/360)
    return colorsys.hsv_to_rgb(v,1,1)
