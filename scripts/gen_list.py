import pdb
import os, glob
# scan the folder
list_im = glob.glob('/data/vision/torralba/scratch2/bzhou/places365/train_large/*/*/*.jpg')
list_im2 = glob.glob('/data/vision/torralba/scratch2/bzhou/places365/train_large/*/*/*/*.jpg')
list_im.extend(list_im2)
list_im.sort()
with open('train.txt','w') as f:
	for line in list_im:
		line = '/'.join(line.split('/')[8:])
		f.write(line + '\n')
