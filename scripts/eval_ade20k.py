import pdb
import sys, glob, socket, time, os, math, random, argparse
import numpy as np
import scipy, h5py

parser = argparse.ArgumentParser()
parser.add_argument('--id', default=0,type=int)
parser.add_argument('--seed', default=3,type=int)
opt = parser.parse_args()
#print(opt)

SEED = opt.seed
ID = opt.id
CAFFE_ROOT = '/data/vision/torralba/segmentation/places/PSPNet/'
MODEL_INFERENCE = 'pspnet50_ADE20K_473.prototxt'
WEIGHTS = '/data/vision/torralba/segmentation/places/PSPNet/evaluation/model/pspnet50_ADE20K.caffemodel'
DEVICE = ID

num_gpus = 4
num_class = 150
base_size = 512
crop_size = 473
stride_rate = 0.6
data_mean = np.array([[[123.68, 116.779, 103.939]]])

# root to image source 
root_image = '/data/vision/oliva/scenedataset/ADE20K_challenge/data/ADEChallengeData2016/images'
txt_imlist = '/data/vision/oliva/scenedataset/ADE20K_challenge/data/ADEChallengeData2016/images/training.txt'

# root to result
root_result = '/data/vision/oliva/scenedataset/scaleplaces/ADE20K/pspnet_prediction'
root_mask = os.path.join(root_result, 'category_mask')
root_prob = os.path.join(root_result, 'prob_mask')
root_maxprob = os.path.join(root_result, 'max_prob')

fn_log = 'logs/%s_seed%d_gpu%d.log'%(socket.gethostname(), SEED, ID)

random.seed(SEED)

# Import Caffe
sys.path.insert(0, os.path.join(CAFFE_ROOT, 'python'))
import caffe
caffe.set_mode_gpu()
caffe.set_device(DEVICE)
net = caffe.Net(MODEL_INFERENCE,
	        WEIGHTS,
	        caffe.TEST)

# read the image list
list_im = [line.rstrip() for line in open(txt_imlist, 'r')]
random.shuffle(list_im)
idx_start = len(list_im)/num_gpus * ID
idx_end = len(list_im)/num_gpus * (ID+1)

cnt = 0
for idx in range(idx_start, idx_end):
	fn_im = list_im[idx]
	cnt += 1
	localtime = time.asctime(time.localtime(time.time()))

	# logging
	if cnt%100==0:		
		with open(fn_log, 'a+') as f_log:
				f_log.write('[%s] Processed [%d]\n' %(localtime, cnt))

	# if output exist, continue
	fn_maxprob = os.path.join(root_maxprob, fn_im.replace('.jpg', '.h5'))
	if os.path.exists(fn_maxprob):
		continue
	# make paths if not exist
	path_mask = os.path.join(root_mask, '/'.join(fn_im.split('/')[0:-1]))
	if not os.path.exists(path_mask):
		os.makedirs(path_mask)
	path_prob = os.path.join(root_prob, '/'.join(fn_im.split('/')[0:-1]))
	if not os.path.exists(path_prob):
		os.makedirs(path_prob)
	path_maxprob = os.path.join(root_maxprob, '/'.join(fn_im.split('/')[0:-1]))
	if not os.path.exists(path_maxprob):
		os.makedirs(path_maxprob)

	# read in image and check its shape
	try:
		image = scipy.misc.imread(os.path.join(root_image, fn_im))
	except:
		with open(fn_log, 'a+') as f_log:
			f_log.write('[%s] Dropped [%s]: bad image\n' %(localtime, fn_im))
		continue

	if image.ndim != 3:
		with open(fn_log, 'a+') as f_log:
			f_log.write('[%s] Modified [%s]: channels != 3\n' %(localtime, fn_im))
		image = np.stack((image,image,image), axis=2)
	
	# resize image
	h_ori = image.shape[0]
	w_ori = image.shape[1]
	if h_ori<128 or w_ori<128:
		with open(fn_log, 'a+') as f_log:
			f_log.write('[%s] Dropped [%s]: image too small\n' %(localtime, fn_im))
		continue
	if w_ori>h_ori:
		image = scipy.misc.imresize(image, (int(1./w_ori*h_ori*base_size), base_size))
	else:
		image = scipy.misc.imresize(image, (base_size, int(1./h_ori*w_ori*base_size)))
	h = image.shape[0]
	w = image.shape[1]

	# substract mean
	image = image.astype('float32') - data_mean
	
	# sliding window params
	stride = crop_size * stride_rate
	h_grid = int(math.ceil(1.*(max(0,h-crop_size))/stride) + 1)
	w_grid = int(math.ceil(1.*(max(0,w-crop_size))/stride) + 1)

	# main loop
	probs = np.zeros((num_class, h, w), dtype=np.float32)
	cnts = np.zeros((1,h,w))
	for h_step in range(h_grid):
		for w_step in range(w_grid):
			# start and end pixel idx
			sh = h_step * stride
			eh = min(h, sh+crop_size)
			sw = w_step * stride
			ew = min(w, sw+crop_size)
			sh, eh, sw, ew = int(sh), int(eh), int(sw), int(ew)

			image_input = np.tile(data_mean, (crop_size, crop_size, 1))
			image_input[0:eh-sh,0:ew-sw,:] = image[sh:eh,sw:ew,:]
			cnts[0,sh:eh,sw:ew] += 1

			# process the image
			net.blobs['data'].data[...] = (image_input[:,:,(2,1,0)].transpose((2,0,1)))[np.newaxis,:,:,:]
			net.forward()
			out = net.blobs['prob'].data[0,:,:,:]
			probs[:,sh:eh,sw:ew] += out[:,0:eh-sh,0:ew-sw]
	
	assert cnts.min()>=1
	probs /= cnts
	assert (probs.min()>=0 and probs.max()<=1), '%f,%f'%(probs.min(),probs.max())
	
	# resize back, it is slow, can I speed it up?
	probs = scipy.ndimage.zoom(probs, (1., 1.*h_ori/h, 1.*w_ori/w), 
		order=1, prefilter=False, mode='nearest')
	
	# calculate output
	pred_mask = np.argmax(probs, axis=0) + 1
	prob_mask = np.max(probs, axis=0)
	max_prob = np.max(probs, axis=(1,2))

	# write to file
	fn_mask = os.path.join(root_mask, fn_im.replace('.jpg', '.png'))
	fn_prob = os.path.join(root_prob, fn_im)
	fn_maxprob = os.path.join(root_maxprob, fn_im.replace('.jpg', '.h5'))

	scipy.misc.imsave(fn_mask, pred_mask.astype('uint8'))
	scipy.misc.imsave(fn_prob, (prob_mask*255).astype('uint8'))
	with h5py.File(fn_maxprob, 'w') as f:
		f.create_dataset('maxprob', data=max_prob)
