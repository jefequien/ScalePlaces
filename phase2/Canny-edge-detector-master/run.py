import os
import argparse
import numpy as np
import random
from scipy import misc

import utils_run as utils
from edge_detector import *

parser = argparse.ArgumentParser()
parser.add_argument("-p", required=True, help="Project name")
parser.add_argument('--start', default=0,type=int)
parser.add_argument("-o", help="Output dir")
args = parser.parse_args()

project = args.p

CONFIG = utils.get_config(project)
im_list = utils.open_im_list(project)

root_images = CONFIG["images"]
root_result = args.o

s = args.start
im_list = im_list[start:start+8000]

for im in im_list:
    print im

    img = misc.imread(argv[1])
    edges = run(img)

    output_path = os.path.join(root_result, im.replace('.jpg', '.png'))

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    misc.imsave(output_path, edges.astype('uint8'))