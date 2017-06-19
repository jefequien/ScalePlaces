import pdb
import sys, glob, time, os
import numpy as np
import scipy, scipy.io
import h5py
from toolkit.toolkit import * 

thres = [0.9, 0.8, 0.7, 0.6]
eval_every = 1000

# root to image source
root_image = '/data/vision/torralba/scratch2/bzhou/places365/train_large'
txt_imlist = 'train.txt'

# root to result
root_result = '/data/vision/torralba/scratch2/hangzhao/scale_places/pspnet_prediction'
root_mask = os.path.join(root_result, 'category_mask')
root_prob = os.path.join(root_result, 'prob_mask')
root_maxprob = os.path.join(root_result, 'max_prob')
root_vis = os.path.join(root_result, 'vis/vis')

fn_log = 'logs/vis.log'

# for vis
fn_color = 'toolkit/color150.mat'
fn_objList = 'toolkit/objectName150.txt'


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
	for key in keys:
		objStr += '<img src="toolkit/color150/%s.jpg">' %(dic[int(key)].split(',')[0])
	return objStr

# read the image list
list_im = [line.rstrip() for line in open(txt_imlist, 'r')]
list_im = list_im[::eval_every]

# Build dictionary: index -> object name
dict_full = objDictFromFile(fn_objList)
colors = scipy.io.loadmat(fn_color)['colors']

# class specific -- change here
for idx_vis in range(150):
	fn_html = os.path.join(root_result, 'vis', str(idx_vis+1) + dict_full[idx_vis+1].split(',')[0] + '.html')
	# ranking output
	list_im_thres = [[] for _ in range(len(thres))]

	# loop to check which thresholding region belongs to
	for i, fn_im in enumerate(list_im):
		if os.path.exists(os.path.join(root_maxprob, fn_im)):
			continue
		localtime = time.asctime( time.localtime(time.time()))

		# read in maxprob
		try:
			with h5py.File(os.path.join(root_maxprob, fn_im.replace('.jpg', '.h5')), 'r') as f:
				maxprob = np.array(f['maxprob'])
		except:
			with open(fn_log, 'a+') as f_log:
				f_log.write('[%s] Dropped [%s]: bad prediction mask\n' %(localtime, fn_im))
			continue

		for j, thre in enumerate(thres):
			if maxprob[idx_vis]>=thre:
				list_im_thres[j].append(fn_im)
				break

	# loop to plot images at different thresholds
	for j, thre in enumerate(thres):
		print 'ID:', idx_vis+1, thre, '#', len(list_im_thres[j])
		with open(fn_html, 'a+') as f:
			f.write('=' * 50 + '<br>'+ '\n')
			f.write('* Max Prob above:  ' + str(thre) + '<br>'+ '\n')
			f.write('=' * 50 + '<br>'+ '\n')

		for fn_im in list_im_thres[j]:
			localtime = time.asctime( time.localtime(time.time()))
			
			f_im = fn_im.split('/',1)[-1]
			f_im = f_im.split('.')[0].replace('/','_')
			fn_mask_color = os.path.join(root_vis, f_im +'_mask_color.png')

			# read in mask
			try:
				image = scipy.misc.imread(os.path.join(root_image, fn_im))
				mask = scipy.misc.imread(os.path.join(root_mask, fn_im.replace('.jpg', '.png')))
				with h5py.File(os.path.join(root_maxprob, fn_im.replace('.jpg', '.h5')), 'r') as f:
					maxprob = np.array(f['maxprob'])
			except:
				with open(fn_log, 'a+') as f_log:
					f_log.write('[%s] Dropped [%s]: bad prediction mask\n' %(localtime, fn_im))
				continue

			mask_unique, mask_ratio = sortUniqueByFreq(mask)

			if not os.path.exists(fn_mask_color):
				# convert pred to color images
				mask_color = colorEncode(mask, colors)

				# Write to images
				scipy.misc.imsave(fn_mask_color, mask_color.astype('uint8'))
		
			# write to html
			with open(fn_html, 'a+') as f:
				f.write('Filename: ' + f_im + '<br>'+ '\n')
				f.write('Max Prob: ' + str(maxprob[idx_vis]) + '<br>'+ '\n')
				f.write(printNames(dict_full, mask_unique) + '<br><br>'+ '\n')
				f.write('<img src="train_large/%s" height="256px"> <img src="vis/%s_mask_color.png" height="256px"><br><br>\n'%(fn_im, f_im))

print 'Done!'
