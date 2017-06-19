import pdb
import os, glob
# scan the folder
list_im = glob.glob('/scratch/bolei/places365_image/val/*/*.jpg')
list_im.sort()
with open('val.txt','w') as f:
	for line in list_im:
		line = '/'.join(line.split('/')[4:])
		f.write(line + '\n')
