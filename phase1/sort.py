import h5py
import numpy as np
import time

import utils

def get_image_paths():
    with open("pspnet_prediction/train.txt", 'r') as f:
        return [line.strip() for line in f.readlines()]

def get_prob_matrix():
    with h5py.File('pspnet_prediction/maxprobs.h5', 'r') as f:
        output = f['maxprobs']
        return output[()]

def get_sorted_matrix():
    with h5py.File('pspnet_prediction/sorted_maxprobs.h5', 'r') as f:
        output = f['sorted_maxprobs']
        return output[()]

def save_sorted_matrix(sorted_indicies_matrix):
    h5f = h5py.File('pspnet_prediction/sorted_maxprobs.h5', 'w')
    h5f.create_dataset('sorted_maxprobs', data=sorted_indicies_matrix)


prob_matrix = get_prob_matrix()
image_paths = get_image_paths()
categories = utils.get_categories()

# sorted_indicies_matrix = np.argsort(prob_matrix, axis=0)
# save_sorted_matrix(sorted_indicies_matrix)
sorted_indicies_matrix = get_sorted_matrix()

print "Done loading."

n,cat = prob_matrix.shape
for c in xrange(1,cat):
    sorted_indicies = sorted_indicies_matrix[:,c][::-1]

    with open("sorted/{}{}.txt".format(c+1,categories[c+1]), 'w') as f:
        for i in sorted_indicies:
            prob = prob_matrix[i,c]
            image_path = image_paths[i]
            f.write("{} {}\n".format(image_path, prob))
            if i % 100000 == 0:
                print categories[c+1], prob