import argparse
import os
import random
import uuid
import time

import utils_vis as utils
from vis_image import ImageVisualizer

class Visualizer:
    def __init__(self, project, output_path=None, MAX=100, special_config=None):
        self.t = time.time()

        self.project = project
        self.image_visualizer = ImageVisualizer(project, special_config=special_config)
        self.images_dir = self.image_visualizer.images_dir

        self.MAX = MAX

        self.output_path = output_path
        if not self.output_path:
            fname = "{}_{}.html".format(project, int(time.time()))
            self.output_path = os.path.join("tmp/", fname)

        if not os.path.exists(os.path.dirname(self.output_path)):
            os.makedirs(os.path.dirname(self.output_path))
        self.init_output_file()
        print time.time() - self.t

    def visualize_images(self, im_list):
        cnt = 0
        for im in im_list[:self.MAX]:
            cnt += 1
            print cnt, im
            self.add_image_section(im)
            print time.time() - self.t

    def init_output_file(self):
        html = "<html><head></head><body></body></html>"
        with open(self.output_path, 'w') as f:
            f.write(html)

        root = "/data/vision/oliva/scenedataset/"
        abs_path = os.path.abspath(self.output_path)
        rel_path = os.path.relpath(abs_path, root)
        print "http://places.csail.mit.edu/{}".format(rel_path)

    def add_image_section(self, im):
        new_section = "<br><br>{} {}<br><br>".format(self.project, im)
        paths = self.image_visualizer.visualize(im)

        order = ["image", "prob_mask", "category_mask", "ground_truth", "diff"]
        for key in order:
            if key in paths:
                new_section += self.getImageTag(paths[key])
                del paths[key]
        # Add the rest
        for key in paths:
            new_section += self.getImageTag(paths[key])

        with open(self.output_path, 'r') as f:
            html = f.read()
        new_html = html.replace("</body>", "{}</body>".format(new_section))
        with open(self.output_path, 'w') as f:
            f.write(new_html)

    def getImageTag(self, path):
        if os.path.isabs(path):
            path = self.symlink(path)

        path = os.path.relpath(path, os.path.dirname(self.output_path))
        return "<img src=\"{}\" height=\"256px\">".format(path)

    def symlink(self, path):
        fn = "{}.jpg".format(uuid.uuid4().hex)
        dst = os.path.join(self.images_dir, fn)
        os.symlink(path, dst)
        return dst

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", required=True, help="Project name")
    parser.add_argument("--imlist", help="Image list")
    parser.add_argument("--pspnet_prediction", help="")
    parser.add_argument("-n", default=10, help="Number of images")
    parser.add_argument('-i',type=int)
    args = parser.parse_args()

    project = args.p
    n = int(args.n)
    
    special_config = None
    if args.pspnet_prediction is not None:
        special_config = utils.get_config(project)
        special_config["pspnet_prediction"] = args.pspnet_prediction

    vis = Visualizer(project, MAX=n, special_config=special_config)

    print "Opening list..."
    im_list = utils.open_im_list(project)
    if args.imlist is not None:
        im_list = utils.open_im_list(args.imlist)

    if args.i:
        im_list = [im_list[args.i]]
    else:
        SEED = 3
        random.seed(SEED)
        #random.shuffle(im_list)

    vis.visualize_images(im_list)

