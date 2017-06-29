import os
import argparse
import numpy as np
import h5py

import utils_eval as utils

def open_data(fname, dataset_name):
    with h5py.File(fname, 'r') as f:
        output = f[dataset_name]
        return output[:]
def eval_data(data, output_file):
    averages = np.nanmean(data, axis=0)
    nonzero = ~np.isnan(data)
    counts = np.sum(nonzero, axis=0)

    output = ""
    categories = utils.get_categories()
    for i in xrange(150):
        avg = averages[i]
        count = counts[i]
        c = i + 1
        output += "{} {} {} {}\n".format(c, categories[c], avg, count)
    with open(output_file, 'w') as f:
        f.write(output)

parser = argparse.ArgumentParser()
parser.add_argument("-f", required=True, help="Date file")
args = parser.parse_args()

fname = args.f
precision = open_data(fname, "precision")
recall = open_data(fname, "recall")
iou = open_data(fname, "iou")
print precision.shape, recall.shape, iou.shape

fname = os.path.basename(fname)
precision_fname = "precision/{}".format(fname.replace(".h5", ".txt"))
eval_data(precision, precision_fname)
recall_fname = "recall/{}".format(fname.replace(".h5", ".txt"))
eval_data(recall, recall_fname)
iou_fname = "iou/{}".format(fname.replace(".h5", ".txt"))
eval_data(iou, iou_fname)
