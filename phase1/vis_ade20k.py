import os
import random

from vis import Visualizer

output_dir = "vis_ade20k"
project = "ade20k"
config = utils.get_data_config(project)

vis = Visualizer(project, output_dir)
for i in xrange(10):
    im_list = config["im_list"]
    random.shuffle(im_list)
    vis.makeHTML(im_list,fname="{}_{}.html".format(project, i))

im_list_dir = "sorted/{}".format(project)
for fname in os.listdir(im_list_dir):
    if ".txt" not in fname:
        continue
    im_list = [line for line in open(os.path.join(im_list_dir,fname), 'r')]
    im_list[:10]
    vis.makeHTML(im_list, fname=txt.replace(".txt", ".html"))
