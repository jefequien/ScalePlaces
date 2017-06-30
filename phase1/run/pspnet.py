import pdb
import sys, glob, socket, time, os, math, random, argparse
import numpy as np
import scipy, h5py

CAFFE_ROOT = '/data/vision/torralba/segmentation/places/PSPNet/'
sys.path.insert(0, os.path.join(CAFFE_ROOT, 'python'))
import caffe

class PSPNet:
    def __init__(self, DEVICE=0):
        caffe.set_mode_gpu()
        caffe.set_device(DEVICE)

        SEED = 3
        random.seed(SEED)

        # MODEL_INFERENCE = 'models/train_pspnet_modified.prototxt'
        MODEL_INFERENCE = 'models/pspnet50_ADE20K_473.prototxt'
        WEIGHTS = '/data/vision/torralba/segmentation/places/PSPNet/evaluation/model/pspnet50_ADE20K.caffemodel'

        self.net = caffe.Net(MODEL_INFERENCE, WEIGHTS, caffe.TEST)
        self.data_mean = np.array([[[123.68, 116.779, 103.939]]])

        self.log = 'logs/%s_seed%d_gpu%d.log'%(socket.gethostname(), SEED, DEVICE)

    def feed_forward(self, image):
        image = image.astype('float32') - self.data_mean

        base_size = 512
        image = utils.resize(image, base_size)

        h,w,n = image.shape
        crop_size = 473
        stride_rate = 0.5
        num_class = 150
        
        # sliding window params
        stride = crop_size * stride_rate
        h_grid = np.arange(0,h,stride)
        w_grid = np.arange(0,w,stride)

        # main loop
        probs = np.zeros((num_class, h, w), dtype=np.float32)
        cnts = np.zeros((1,h,w))
        for sh in h_grid:
            for sw in w_grid:
                # start and end pixel idx
                eh = min(h, sh+crop_size)
                ew = min(w, sw+crop_size)
                sh, eh, sw, ew = int(sh), int(eh), int(sw), int(ew)

                image_input = np.tile(self.data_mean, (crop_size, crop_size, 1))
                image_input[0:eh-sh,0:ew-sw,:] = image[sh:eh,sw:ew,:]
                cnts[0,sh:eh,sw:ew] += 1

                # process the image
                self.net.blobs['data'].data[...] = (image_input[:,:,(2,1,0)].transpose((2,0,1)))[np.newaxis,:,:,:]
                net.forward()
                out = net.blobs['prob'].data[0,:,:,:]
                probs[:,sh:eh,sw:ew] += out[:,0:eh-sh,0:ew-sw]
        
        assert cnts.min()>=1
        probs /= cnts
        assert (probs.min()>=0 and probs.max()<=1), '%f,%f'%(probs.min(),probs.max())
        

    def get_network_architecture(self):
        for k,v in self.net.blobs.items():
            print v.data.shape, k

pspnet = PSPNet()
pspnet.get_network_architecture()
