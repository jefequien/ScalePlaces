import time
import numpy as np
import itertools
from scipy import misc, ndimage

DATA_MEAN = np.array([[[123.68, 116.779, 103.939]]])
INPUT_SIZE = 473
NUM_CLASS = 150

def process(self, image):
    localtime = time.asctime(time.localtime(time.time()))

    if image.ndim != 3:
        with open(self.log, 'a+') as f:
            f.write('[%s] Modified image: channels != 3\n' %(localtime))
        image = np.stack((image,image,image), axis=2)

    image = image.astype('float32') - DATA_MEAN
    
    # Resize consistent size
    h_ori,w_ori,n = image.shape
    short_side = min(h_ori, w_ori)
    long_side = max(h_ori, w_ori)
    ratio = 1.*512/long_side
    image = misc.imresize(image, ratio)

    h,w,n = image.shape
    stride_rate = 0.3
    stride = INPUT_SIZE * stride_rate
    hs_upper = max(1,h-(INPUT_SIZE-stride))
    ws_upper = max(1,w-(INPUT_SIZE-stride))
    hs = np.arange(0,hs_upper,stride, dtype=int)
    ws = np.arange(0,ws_upper,stride, dtype=int)
    locs = list(itertools.product(hs,ws))
    #print image.shape
    #print hs
    #print ws

    probs = np.zeros((NUM_CLASS, h, w), dtype=np.float32)
    cnts = np.zeros((1,h,w))
    for loc in locs:
        sh,sw = loc
        eh = min(h, sh + INPUT_SIZE)
        ew = min(w, sw + INPUT_SIZE)
        
        data = np.tile(DATA_MEAN, (INPUT_SIZE, INPUT_SIZE, 1))
        data[0:eh-sh,0:ew-sw,:] = image[sh:eh,sw:ew,:]

        out = self.feed_forward(data)

        cnts[0,sh:eh,sw:ew] += 1
        probs[:,sh:eh,sw:ew] += out[:,0:eh-sh,0:ew-sw]

    assert cnts.min()>=1
    probs /= cnts
    assert (probs.min()>=0 and probs.max()<=1), '%f,%f'%(probs.min(),probs.max())

    # Resize back
    probs = ndimage.zoom(probs, (1.,1.*h_ori/h,1.*w_ori/w), order=1, prefilter=False, mode='nearest')
    assert probs.shape == (NUM_CLASS,h_ori,w_ori)
    return probs
