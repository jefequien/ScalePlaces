import os
import sys
sys.path.append(os.path.abspath('../utils'))
from utils import *

def resize(data, base_size, interp="bilinear"):
    h_ori = data.shape[0]
    w_ori = data.shape[1]

    shape = None
    if w_ori < h_ori:
        shape = (int(1./w_ori*h_ori*base_size), base_size)
    else:
        shape = (base_size, int(1./h_ori*w_ori*base_size))
    resized = misc.imresize(data, shape, interp=interp)
    return resized