import pdb
import sys, glob, socket, time, os, math, random, argparse
import numpy as np
import scipy, h5py

CAFFE_ROOT = '/data/vision/torralba/segmentation/places/PSPNet/'
sys.path.insert(0, os.path.join(CAFFE_ROOT, 'python'))
import caffe

class PSPNet:
    def __init__(self):

        DEVICE = 0
        SEED = 3
        caffe.set_mode_gpu()
        caffe.set_device(DEVICE)

        MODEL_INFERENCE = 'pspnet50_ADE20K_473.prototxt'
        WEIGHTS = '/data/vision/torralba/segmentation/places/PSPNet/evaluation/model/pspnet50_ADE20K.caffemodel'
        self.net = caffe.Net(MODEL_INFERENCE, WEIGHTS, caffe.TEST)

        random.seed(SEED)

        fn_log = 'logs/%s_seed%d_gpu%d.log'%(socket.gethostname(), SEED, DEVICE)

    def process(self, image):
        pass
    def get_network_architecture(self):
        layernames = self.net.blob_names
        print layernames
        for layername in layernames:
            print self.net.blobs(layername).shape



# parser = argparse.ArgumentParser()
# parser.add_argument('--id', default=0,type=int)
# parser.add_argument('--seed', default=3,type=int)
# opt = parser.parse_args()
# #print(opt)


# num_gpus = 1
# num_class = 150
# base_size = 512
# crop_size = 473
# stride_rate = 0.6
# data_mean = np.array([[[123.68, 116.779, 103.939]]])

# # root to image source 
# root_image = ''
# txt_imlist = ''

# # root to result
# root_result = ''



# # read the image list
# list_im = [line.rstrip() for line in open(txt_imlist, 'r')]
# # random.shuffle(list_im)

# for i in range(0, len(list_im)):
#     fn_im = list_im[i]
    
#     # resize image
#     h_ori = image.shape[0]
#     w_ori = image.shape[1]
#     if h_ori<128 or w_ori<128:
#         with open(fn_log, 'a+') as f_log:
#             f_log.write('[%s] Dropped [%s]: image too small\n' %(localtime, fn_im))
#         continue
#     if w_ori>h_ori:
#         image = scipy.misc.imresize(image, (int(1./w_ori*h_ori*base_size), base_size))
#     else:
#         image = scipy.misc.imresize(image, (base_size, int(1./h_ori*w_ori*base_size)))
#     h = image.shape[0]
#     w = image.shape[1]
#     # substract mean
#     image = image.astype('float32') - data_mean
    
#     # sliding window params
#     stride = crop_size * stride_rate
#     h_grid = int(math.ceil(1.*(max(0,h-crop_size))/stride) + 1)
#     w_grid = int(math.ceil(1.*(max(0,w-crop_size))/stride) + 1)

#     # main loop
#     probs = np.zeros((num_class, h, w), dtype=np.float32)
#     cnts = np.zeros((1,h,w))
#     for h_step in range(h_grid):
#         for w_step in range(w_grid):
#             # start and end pixel idx
#             sh = h_step * stride
#             eh = min(h, sh+crop_size)
#             sw = w_step * stride
#             ew = min(w, sw+crop_size)
#             sh, eh, sw, ew = int(sh), int(eh), int(sw), int(ew)

#             image_input = np.tile(data_mean, (crop_size, crop_size, 1))
#             image_input[0:eh-sh,0:ew-sw,:] = image[sh:eh,sw:ew,:]
#             cnts[0,sh:eh,sw:ew] += 1

#             # process the image
#             net.blobs['data'].data[...] = (image_input[:,:,(2,1,0)].transpose((2,0,1)))[np.newaxis,:,:,:]
#             net.forward()
#             out = net.blobs['prob'].data[0,:,:,:]
#             probs[:,sh:eh,sw:ew] += out[:,0:eh-sh,0:ew-sw]
    
#     assert cnts.min()>=1
#     probs /= cnts
#     assert (probs.min()>=0 and probs.max()<=1), '%f,%f'%(probs.min(),probs.max())
    
#     # resize back, it is slow, can I speed it up?
#     probs = scipy.ndimage.zoom(probs, (1., 1.*h_ori/h, 1.*w_ori/w), 
#         order=1, prefilter=False, mode='nearest')
    

