import pdb
import os
import numpy as np
import h5py

root_result = '/data/vision/torralba/scratch2/hangzhao/scale_places/pspnet_prediction'
root_maxprob = os.path.join(root_result, 'max_prob')

txt_imlist = 'train.txt'
list_im = [line.rstrip() for line in open(txt_imlist, 'r')]
maxprobs = np.zeros((len(list_im), 150), dtype=np.float32)

for i, fn_im in enumerate(list_im):
	fn_maxprob = os.path.join(root_maxprob, fn_im.replace('.jpg', '.h5'))
	with h5py.File(fn_maxprob, 'r') as f:
		maxprobs[i] = f['maxprob']

fn_out = os.path.join(root_result, 'maxprobs.h5')
with h5py.File(fn_out, 'w') as f:
	f.create_dataset('maxprobs', data=maxprobs)

pdb.set_trace()
