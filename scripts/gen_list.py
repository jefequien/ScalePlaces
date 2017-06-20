import pdb
import os, glob
# scan the folder
list_im = glob.glob('/data/vision/oliva/scenedataset/ADE20K_challenge/data/ADEChallengeData2016/images/validation/*.jpg')
list_im.sort()
with open('ade20k_val.txt','w') as f:
	for line in list_im:
		line = '/'.join(line.split('/')[10:])
		f.write(line + '\n')
