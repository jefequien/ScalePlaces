import os
import sys
sys.path.append(os.path.abspath('../utils'))
from utils import *

def resize(image, base_size, interp="bilinear"):
    h_ori = image.shape[0]
    w_ori = image.shape[1]

    shape = None
    if w_ori < h_ori:
        shape = (int(1./w_ori*h_ori*base_size), base_size)
    else:
        shape = (base_size, int(1./h_ori*w_ori*base_size))
    resized = misc.imresize(image, shape, interp=interp)
    return resized

def zoom(data, shape):
    n,h,w = data.shape
    ratios = (1., 1.*shape[0]/h, 1.*shape[1]/w)
    data = scipy.ndimage.zoom(probs, ratios, order=1, prefilter=False, mode='nearest')