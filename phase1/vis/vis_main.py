import os
import random

import utils_vis as utils
from vis import Visualizer

project = "street_view"
output_dir = "vis_{}".format(project)
config = utils.get_data_config(project)
im_list = utils.open_im_list(project)

vis = Visualizer(project, output_dir, MAX=10)
for i in xrange(1):
    #random.shuffle(im_list)
    vis.makeHTML(im_list,fname="{}_{}.html".format(project,i))

raise
im_list_dir = "../eval/sorted/{}".format(project)
for fname in os.listdir(im_list_dir):
    if ".txt" not in fname:
        continue
    im_list = [line for line in open(os.path.join(im_list_dir,fname), 'r')]
    vis.makeHTML(im_list, fname=fname.replace(".txt", ".html"))
