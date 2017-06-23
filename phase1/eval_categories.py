import os
import argparse
import numpy as np
import h5py

import utils

def open_data(fname, dataset_name):
    with h5py.File(fname, 'r') as f:
        output = f[dataset_name]
        return output[:]


parser = argparse.ArgumentParser()
parser.add_argument("-f", required=True, help="Date file")
args = parser.parse_args()

fname = args.f
data = open_data(fname, "accuracies")
averages = np.nanmean(data, axis=0)

categories = utils.get_categories()
for i in xrange(len(averages)):
    avg = averages[i]
    #avg = "%.3f" % avg
    c = i+1
    output = "{} {} {}".format(c, categories[c], avg)
    print output
