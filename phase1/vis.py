import argparse
import os
import uuid

from scipy import misc
import numpy as np

import utils
import time
import random

class Visualizer:
    def __init__(self, output_dir="tmp/"):
        self.output_dir = output_dir
        self.images_dir = "tmp/images/"
        if not os.path.exists(self.images_dir):
            os.makedirs(self.images_dir)

        self.max = 100

    def makeImageSection(self, project, im):
        html = "{} {}<br><br>".format(project, im)

        # Paths
        config = utils.get_data_config(project)
        root_images = config["images"]
        root_category_mask = os.path.join(config["pspnet_prediction"], "category_mask")
        root_prob_mask = os.path.join(config["pspnet_prediction"], "prob_mask")
        root_ground_truth = config["ground_truth"]

        image = os.path.join(root_images, im)
        category_mask = os.path.join(root_category_mask, im.replace('.jpg','.png'))
        prob_mask = os.path.join(root_prob_mask, im)

        try:
            color_mask = self.createColorMask(category_mask)
        except:
            color_mask = None
        try:
            gt_path = os.path.join(root_ground_truth, im.replace('.jpg','.png'))
            ground_truth = self.createColorMask(gt_path)
        except:
            ground_truth = None
        try:
            gt_path = os.path.join(root_ground_truth, im.replace('.jpg','.png'))
            diff = self.createDiffImage(category_mask, gt_path)
            error = self.createColorMask(diff)
        except:
            error = None

        paths = [image,prob_mask,color_mask, ground_truth, error]
        for path in paths:
            html += self.getImageTag(path)

        html += "<br><br>"
        return html

    def makeHTML(self, project, im_list, fname=None):
        sections = ""
        cnt = 0
        for im in im_list:
            print cnt, im
            sections += self.makeImageSection(project, im)

            cnt += 1
            if cnt == self.max:
                break

        body = "<body>{}</body>".format(sections)
        html = "<html><head></head>{}</html>".format(body)

        if not fname:
            fname = "{}_{}.html".format(project, int(time.time()))

        output_path = os.path.join(self.output_dir,fname)
        with open(output_path, 'w') as f:
            f.write(html)
        return output_path

    def createDiffImage(self, path, gt_path):
        pred = misc.imread(path)
        gt = misc.imread(gt_path)
        gt[gt==pred] = 0
        return gt

    def createColorMask(self, category_mask_path):
        category_mask = misc.imread(category_mask_path)
        h,w = category_mask.shape
        color_mask = np.zeros((h,w,3))
        for i in xrange(1,151):
            color_mask[category_mask == i] = utils.to_color(i)

        fn = "{}.jpg".format(uuid.uuid4().hex)
        path = os.path.join(self.images_dir, fn)
        misc.imsave(path, color_mask)
        return path

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
    parser.add_argument("-i", help="Image name")
    parser.add_argument("--imlist", help="Image list")
    parser.add_argument("--output", help="Output directory")
    args = parser.parse_args()

    project = args.p
    im = args.i
    im_list_file = args.imlist
    output_dir = args.output

    im_list = []
    if im:
        im_list = [im]
    elif im_list_file:
        im_list = [line.split()[0] for line in open(im_list_file, 'r')]
    else:
        config = utils.get_data_config(project)
        im_list_file = config["im_list"]
        if im_list_file:
            im_list = [line.rstrip() for line in open(im_list_file, 'r')]
            random.shuffle(im_list)

    vis = Visualizer()
    if output_dir:
        vis = Visualizer(output_dir=output_dir)

    output_path = vis.makeHTML(project, im_list)
    print "http://places.csail.mit.edu/scaleplaces/ScalePlaces/phase1/{}".format(output_path)

