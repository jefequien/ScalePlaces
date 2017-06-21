import argparse
import os
import random
import uuid
import time

import utils
from vis_image import ImageVisualizer

class Visualizer:
    def __init__(self, project, output_dir, MAX=1000):
        self.project = project
        self.image_visualizer = ImageVisualizer(project)
        self.images_dir = self.image_visualizer.images_dir

        self.output_dir = output_dir
        self.MAX = MAX

    def makeHTML(self, im_list, fname=None):
        img_sections = ""
        cnt = 0
        for im in im_list[:self.MAX]:
            img_section = "<br><br>{} {}<br><br>".format(self.project, im)

            im = im.split()[0]
            paths = self.image_visualizer.visualize(im)
            img_section += self.getImageTag(paths["image"])
            img_section += self.getImageTag(paths["prob_mask"])
            img_section += self.getImageTag(paths["category_mask"])
            img_section += self.getImageTag(paths["ground_truth"])
            img_section += self.getImageTag(paths["diff"])

            img_sections += img_section

            print cnt, im
            cnt += 1

        body = "<body>{}</body>".format(img_sections)
        html = "<html><head></head>{}</html>".format(body)

        if not fname:
            fname = "{}_{}.html".format(project, int(time.time()))

        output_path = os.path.join(self.output_dir,fname)
        with open(output_path, 'w') as f:
            f.write(html)
        return output_path

    def getImageTag(self, path):
        if path:
            if os.path.isabs(path):
                path = self.symlink(path)
            path = os.path.relpath(path, self.output_dir)
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
    parser.add_argument("-n", default=100, help="Number of images")
    parser.add_argument("-o", default="tmp/", help="Output directory")
    args = parser.parse_args()

    project = args.p
    n = int(args.n)
    output_dir = args.o

    im_list = []
    if args.imlist:
        im_list = [line for line in open(args.imlist, 'r')]
    else:
        config = utils.get_data_config(project)
        im_list = [line for line in open(config["im_list"], 'r')]
        random.shuffle(im_list)

    vis = Visualizer(project, output_dir, MAX=n)

    output_path = vis.makeHTML(im_list)
    print "http://places.csail.mit.edu/scaleplaces/ScalePlaces/phase1/{}".format(output_path)

