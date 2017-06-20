import h5py
import numpy as np
import os

import utils

def build_maxprobs(im_list, maxprob_root):
    maxprobs = np.empty((0,150), float)
    for im in im_list:
        with h5py.File(os.path.join(maxprob_root, im), 'r') as f:
            data = ['maxprob'][()]
            np.append(maxprobs, data, axis=0)
    with h5py.File("maxprobs.h5", 'w') as f:
        f.create_dataset('maxprobs', data=maxprobs)
    return maxprobs

def open_maxprobs():
    with h5py.File('maxprobs.h5', 'r') as f:
        output = f['maxprobs']
        return output[()]

def sort_maxprobs(maxprobs):
    sorted_maxprobs = np.argsort(maxprobs, axis=0)
    with h5py.File("sorted_maxprobs.h5", 'w') as f:
        f.create_dataset('sorted_maxprobs', data=maxprobs)
        return output[()]

def open_sorted_maxprobs():
    with h5py.File('sorted_maxprobs.h5', 'r') as f:
        output = f['sorted_maxprobs']
        return output[()]


project = "ade20k"
config = utils.get_data_config(project)

im_list_path = config["im_list"]
im_list = [line.rstrip() for line in open(im_list_path, 'r')]

pspnet_pred_root = config["pspnet_prediction"]
maxprob_root = os.path.join(pspnet_pred_root, "maxprob")
maxprobs = build_maxprobs(im_list, maxprob_root)
# maxprobs = open_maxprobs()

sorted_maxprobs = sort_maxprobs(maxprobs)
# sorted_maxprobs = open_sorted_maxprobs()

print "Done loading."

categories = utils.get_categories()
n,cat = maxprobs.shape
for c in xrange(0,cat):
    sorted_indicies = sorted_indicies_matrix[:,c][::-1]

    with open("sorted/{}/{}{}.txt".format(project, c+1, categories[c+1]), 'w') as f:
        for i in sorted_indicies:
            prob = maxprobs[i,c]
            im = im_list[i]
            f.write("{} {}\n".format(im, prob))
            if i % 100000 == 0:
                print categories[c+1], prob