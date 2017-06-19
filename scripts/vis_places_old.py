import pdb
import sys, glob, time, os
import numpy as np
import matplotlib.pyplot as plt
import scipy, scipy.io
from toolkit.toolkit import * 

CAFFE_ROOT = '/data/vision/torralba/segmentation/places/PSPNet/'
MODEL_INFERENCE =  '/data/vision/torralba/segmentation/places/PSPNet/evaluation/prototxt/pspnet50_ADE20K_473.prototxt'
WEIGHTS = '/data/vision/torralba/segmentation/places/PSPNet/evaluation/model/pspnet50_ADE20K.caffemodel'
DEVICE = 2

num_class = 150
size_im = (473, 473)
eval_every = 250

path_source_image = '/data/vision/oliva/scenedataset/places2new/challenge2016'
path_result = '/data/vision/torralba/segmentation/places/result/PSPNet'
path_result_meta = path_result

txt_imlist = os.path.join(path_source_image, 'places365_train_max5000_filenamelist_standard_large.txt')
fn_color = 'toolkit/color150.mat'
fn_objList = 'toolkit/objectName150.txt'
fn_html = '/data/vision/torralba/segmentation/places/result/PSPNet.html'
fn_log = '/data/vision/torralba/segmentation/places/result/PSPNet_log.txt'

def getClassNameAccu(dic, keys, ratio):
	names = ''
	accus = ''
	for i in range(len(keys)):
		if (keys[i]==0) or (ratio[i]*100<0.5):
			continue
		names += '"%s",' %(dic[int(keys[i])].split(',')[0])
		accus += '"%.2f%%",' %(ratio[i]*100)
	return names[:-1], accus[:-1]

def printNames(dic, keys):
	objStr = ''
	for i in range(len(keys)):
		objStr += '<img src="toolkit/color150/%s.jpg">' %(dic[int(keys[i])].split(',')[0])
	return objStr

if __name__=='__main__':
	# Check files and folders
	assert os.path.isfile(MODEL_INFERENCE), 'Missing model file!'
	assert os.path.isfile(WEIGHTS), 'Missing weight file!'

	if not os.path.exists(path_result):
		os.makedirs(path_result)
	if not os.path.exists(path_result_meta):
		os.makedirs(path_result_meta)

	# Import Caffe
	sys.path.insert(0, os.path.join(CAFFE_ROOT, 'python'))
	import caffe
	caffe.set_mode_gpu()
	caffe.set_device(DEVICE)
	net = caffe.Net(MODEL_INFERENCE,
		        WEIGHTS,
		        caffe.TEST)

	# Build dictionary: index -> object name
	dict_full = objDictFromFile(fn_objList)

	colors = scipy.io.loadmat(fn_color)['colors']

	# scan the folder
	# list_im = glob.glob(path_source_image + '*.jpg') + glob.glob(path_source_image + '*.png')
	# read the image list
	list_im = []
	with open(txt_imlist, 'r') as f:
		for line in f:
			list_im.append(line.rstrip().split(' ')[0])

	cnt = 0
	for i, fn_im in enumerate(list_im):
		if i%eval_every!=0:
			continue
		if cnt%100==0:
			print 'Finished: %d'%(cnt)

		localtime = time.asctime( time.localtime(time.time()))

		# read in image and check its shape
		try:
			image = scipy.misc.imread(os.path.join(path_source_image, fn_im))
		except:
			with open(fn_log, 'a+') as f_log:
				f_log.write('[%s] Dropped [%s]: bad image\n' %(localtime, fn_im))
			continue

		if image.ndim != 3:
			with open(fn_log, 'a+') as f_log:
				f_log.write('[%s] Modified [%s]: channels != 3\n' %(localtime, fn_im))
			image = np.stack((image,image,image), axis=2)
		
		h = image.shape[0]
		w = image.shape[1]
		if h<128 or w<128:
			with open(fn_log, 'a+') as f_log:
				f_log.write('[%s] Dropped [%s]: image too small\n' %(localtime, fn_im))
			continue

		# resize image
		image = scipy.misc.imresize(image, size_im)
		# substract mean
		image = image.astype('float32')-np.array([[[123.68, 116.779, 103.939]]])
		
		# process the image
		logits = np.zeros((num_class, size_im[0], size_im[1]))
		net.blobs['data'].data[...] = (image[:,:,(2,1,0)].transpose((2,0,1)))[np.newaxis,:,:,:]
		net.forward()
		logits += np.squeeze(net.blobs['conv6_interp'].data[0,:,:,:])
		net.blobs['data'].data[...] = (image[:,::-1,(2,1,0)].transpose((2,0,1)))[np.newaxis,:,:,:]
		net.forward()
		logits += np.squeeze(net.blobs['conv6_interp'].data[0,:,:,:])[:,:,::-1]

		pred = np.argmax(logits, axis=0) + 1

		pred_unique, pred_ratio = sortUniqueByFreq(pred)
		pred = scipy.misc.imresize(pred.astype('uint8'), (h,w), interp='nearest')
		
		# convert pred to color images
		pred_color = colorEncode(pred, colors)

		# Write to images
		f_im = fn_im.split('/',1)[-1]
		f_im = f_im.split('.')[0].replace('/','_')
		# scipy.misc.imsave(os.path.join(path_result, f_im +'.jpg'), image.astype('uint8'))
		fn_pred_color = os.path.join(path_result, f_im +'_pred_color.png')
		fn_pred_gray = os.path.join(path_result, f_im +'_pred_gray.png')
		scipy.misc.imsave(fn_pred_color, pred_color.astype('uint8'))
		scipy.misc.imsave(fn_pred_gray, pred.astype('uint8'))

		names, ratios = getClassNameAccu(dict_full, pred_unique, pred_ratio)
		# Write metadata (JSON) to csv
		with open(os.path.join(path_result_meta, f_im + '.csv'), 'w') as f: 

			curLine = '{"object":{'
			curLine += '"segment":"%s",\n' %(f_im +'_pred_color.png')
			curLine += '"classes": [%s],\n' %(names)
			curLine += '"ratios": [%s]\n' %(ratios)
			curLine += '}}'

			f.write(curLine)

		# write to html
		with open(fn_html, 'a+') as f:
			f.write('Filename: ' + f_im + '<br>'+ '\n')
			f.write(printNames(dict_full, pred_unique) + '<br><br>'+ '\n')
			f.write('<img src="%s" height="256px"> <img src="PSPNet/%s_pred_color.png" height="256px"><br><br>\n'%(fn_im, f_im))

		cnt += 1
		# with open(fn_log, 'a+') as f_log:
		# 		f_log.write('[%s] Processed [%s]\n' %(localtime, fn_im))

	print 'Done!'