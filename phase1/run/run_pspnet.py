import os
import os
import argparse
import numpy as np
import h5py
import random
from scipy import misc

from pspnet import PSPNet
import utils_run as utils

parser = argparse.ArgumentParser()
parser.add_argument("-p", required=True, help="Project name")
parser.add_argument("--model", required=True, help="Model")
parser.add_argument("--snapshot", required=True, help="Snapshot .caffemodel")
parser.add_argument('--id', default=0,type=int)
parser.add_argument('--local', action='store_true', default=False, help="./predictions")
args = parser.parse_args()

project = args.p
model = args.model
snapshot = args.snapshot
pspnet = PSPNet(model, snapshot, DEVICE=args.id)

CONFIG = utils.get_config(project)
im_list = utils.open_im_list(project)

root_images = CONFIG["images"]
root_result = CONFIG["pspnet_prediction"]
if args.local:
    root_result = snapshot.replace("snapshots", "predictions/{}".format(project))
    root_result = root_result.replace(".caffemodel", "/")
print "Outputting to ", root_result

root_mask = os.path.join(root_result, 'category_mask')
root_prob = os.path.join(root_result, 'prob_mask')
root_maxprob = os.path.join(root_result, 'max_prob')
root_allprob = os.path.join(root_result, 'all_prob')

random.seed(3)
random.shuffle(im_list)
for im in im_list:
    print im

    fn_maxprob = os.path.join(root_maxprob, im.replace('.jpg', '.h5'))
    fn_mask = os.path.join(root_mask, im.replace('.jpg', '.png'))
    fn_prob = os.path.join(root_prob, im)
    fn_allprob = os.path.join(root_allprob, im.replace('.jpg', '.h5'))

    if os.path.exists(fn_maxprob):
        print "Already done."
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
        image = utils.get_file(im, CONFIG, ftype="im")
    except:
        print "Unable to load image. Skipping..."
        continue

    probs = pspnet.sliding_window(image)
    # probs is 150 x h x w

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
