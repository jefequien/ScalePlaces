import os
import argparse
import numpy as np
from scipy import misc
import h5py

import utils

def evaluate_image(im):
    # cm = misc.imread(os.path.join(root_cm, im.replace(".jpg",".png")))
    gt = misc.imread(os.path.join(root_gt, im.replace(".jpg",".png")))
    squish = misc.imresize(gt, 60.0/473, interp='nearest')
    cm = misc.imresize(squish, gt.shape, interp='nearest')

    accuracy = {}
    for c in xrange(1,151):
        cm_mask = cm == c
        gt_mask = gt == c
        intersection = np.logical_and(cm_mask, gt_mask)
        union = np.logical_or(cm_mask, gt_mask)

        if np.sum(union) != 0:
            iou = 1.0*np.sum(intersection)/np.sum(union)
            gt_area = np.sum(gt_mask)
            accuracy[c] = iou
    return accuracy

def evaluate_images(im_list):
    n = len(im_list)
    accuracies = np.zeros((n, 150))
    for i in xrange(n):
        im = im_list[i]
        print im

        try:
            accuracy = evaluate_image(im)
        except:
            accuracy = {}
            print "Skipping", im

        for c in xrange(1,151):
            if c in accuracy:
                accuracies[i,c-1] = accuracy[c]
            else:
                accuracies[i,c-1] = np.nan
    return accuracies

parser = argparse.ArgumentParser()
parser.add_argument("-p", required=True, help="Project name")
args = parser.parse_args()

project = args.p
config = utils.get_data_config(project)
root_images = config["images"]
root_cm = os.path.join(config["pspnet_prediction"], "category_mask")
root_pm = os.path.join(config["pspnet_prediction"], "prob_mask")
root_gt = config["ground_truth"]

im_list = [line.rstrip() for line in open(config["im_list"], 'r')]
#im_list = im_list[:100]
accuracies = evaluate_images(im_list)
print accuracies.shape

fname = "{}_acc_squish.h5".format(project)
with h5py.File(fname, 'w') as f:
    f.create_dataset('accuracies', data=accuracies)
