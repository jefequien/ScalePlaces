import os
import argparse
import numpy as np
import h5py
from scipy import misc

from pspnet import PSPNet
import utils_run as utils


root_result = 'my_pspnet_prediction'
root_mask = os.path.join(root_result, 'category_mask')
root_prob = os.path.join(root_result, 'prob_mask')
root_maxprob = os.path.join(root_result, 'max_prob')
root_allprob = os.path.join(root_result, 'all_prob')

parser = argparse.ArgumentParser()
parser.add_argument('--id', default=0,type=int)
opt = parser.parse_args()

pspnet = PSPNet(DEVICE=opt.id)
pspnet.get_network_architecture()

project = "street_view"
CONFIG = utils.get_data_config(project)
im_list = utils.open_im_list(project)
root_images = CONFIG["images"]

for fn_im in im_list:
    print fn_im

    fn_maxprob = os.path.join(root_maxprob, fn_im.replace('.jpg', '.h5'))
    fn_mask = os.path.join(root_mask, fn_im.replace('.jpg', '.png'))
    fn_prob = os.path.join(root_prob, fn_im)
    fn_allprob = os.path.join(root_allprob, fn_im.replace('.jpg', '.h5'))

    if os.path.exists(fn_maxprob):
        continue

    # make paths if not exist
    if not os.path.exists(os.path.dirname(fn_maxprob)):
        os.makedirs(os.path.dirname(fn_maxprob))
    if not os.path.exists(os.path.dirname(fn_mask)):
        os.makedirs(os.path.dirname(fn_mask))
    if not os.path.exists(os.path.dirname(fn_prob)):
        os.makedirs(os.path.dirname(fn_prob))
    if not os.path.exists(os.path.dirname(fn_allprob)):
        os.makedirs(os.path.dirname(fn_allprob))

    try:
        image = utils.get_file(fn_im, CONFIG, ftype="im")
    except:
        print "Bad image"
        continue
    probs = pspnet.process(image)

    # calculate output
    pred_mask = np.argmax(probs, axis=0) + 1
    prob_mask = np.max(probs, axis=0)
    max_prob = np.max(probs, axis=(1,2))
    all_prob = probs

    # write to file
    misc.imsave(fn_mask, pred_mask.astype('uint8'))
    misc.imsave(fn_prob, (prob_mask*255).astype('uint8'))
    with h5py.File(fn_maxprob, 'w') as f:
        f.create_dataset('maxprob', data=max_prob)
    with h5py.File(fn_allprob, 'w') as f:
        f.create_dataset('allprob', data=all_prob)
