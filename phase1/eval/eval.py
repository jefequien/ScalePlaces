import os
import argparse
import numpy as np
from scipy import misc
import h5py
import traceback

import utils_eval as utils

def evaluate_image(im, threshold=0.5):
    try:
        cm = utils.get(im, CONFIG, ftype="cm")
        ap = utils.get(im, CONFIG, ftype="ap")
        gt = utils.get(im, CONFIG, ftype="gt")
    except:
        return None

    # precision, recall, iou
    results = np.zeros((150,3))
    results[:] = np.nan
    for i in xrange(150):
        c = i+1
        probs = ap[i,:,:]
        prob_mask = probs > threshold
        cm_mask = cm == c
        gt_mask = gt == c
        
        mask = prob_mask

        intersection = np.logical_and(mask, gt_mask)
        union = np.logical_or(mask, gt_mask)
        if np.sum(prob_mask) != 0:
            precision = 1.*np.sum(intersection)/np.sum(prob_mask)
            results[i,0] = precision
        else:
            results[i,0] = np.nan
        if np.sum(gt_mask) != 0:
            recall = 1.*np.sum(intersection)/np.sum(gt_mask)
            results[i,1] = recall
        else:
            results[i,1] = np.nan
        if np.sum(union) != 0:
            iou = 1.0*np.sum(intersection)/np.sum(union)
            results[i,2] = iou
        else:
            results[i,2] = np.nan
    return results

def evaluate_images(im_list):
    n = len(im_list)
    all_results = np.zeros((n, 150, 3))
    for i in xrange(n):
        im = im_list[i]
        print im

        results = evaluate_image(im, threshold=0.5)
        if results is not None:
            all_results[i] = results
        else:
            print "Skipping", im
            all_results[i,:,:] = np.nan
    return all_results

parser = argparse.ArgumentParser()
parser.add_argument("-p", required=True, help="Project name")
args = parser.parse_args()

project = args.p
CONFIG = utils.get_data_config(project)

im_list = [line.rstrip() for line in open(CONFIG["im_list"], 'r')]
#im_list = im_list[:100]
results = evaluate_images(im_list)
print results.shape

fname = "{}_eval.h5".format(project)
with h5py.File(fname, 'w') as f:
    f.create_dataset('precision', data=results[:,:,0])
    f.create_dataset('recall', data=results[:,:,1])
    f.create_dataset('iou', data=results[:,:,2])
