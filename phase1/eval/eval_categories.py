import os
import argparse
import numpy as np
import h5py

import utils_eval as utils

def open_data(fname, dataset_name):
    with h5py.File(fname, 'r') as f:
        output = f[dataset_name]
        return output[:]
def eval_data(data):
    averages = np.nanmean(data, axis=0)
    nonzero = ~np.isnan(data)
    counts = np.sum(nonzero, axis=0)

    categories = utils.get_categories()
    for i in xrange(150):
        avg = averages[i]
        count = counts[i]
        c = i + 1
        output = "{} {} {} {}".format(c, categories[c], avg, count)
        print output

parser = argparse.ArgumentParser()
parser.add_argument("-f", required=True, help="Date file")
args = parser.parse_args()

fname = args.f
precision = open_data(fname, "precision")
recall = open_data(fname, "recall")
iou = open_data(fname, "iou")
print precision.shape, recall.shape, iou.shape

eval_data(iou)

