import os
import os
import argparse
import numpy as np
import h5py
import random
from scipy import misc

from data_source import DataSource
from refine_net import RefineNet
import utils_run as utils

def build_output_dirname(project, snapshot):
    template = "predictions/{}/{}/"
    more = snapshot.replace("snapshots/", "")
    more = more.replace(".caffemodel", "")
    return template.format(project, more)

parser = argparse.ArgumentParser()
parser.add_argument("-p", required=True, help="Project name")
parser.add_argument("--model", required=True, help="Model")
parser.add_argument("--snapshot", required=True, help="Snapshot .caffemodel")
parser.add_argument("--pspnet_prediction", required=True, help="Source pspnet prediction")
parser.add_argument('--id', default=0,type=int)
args = parser.parse_args()

project = args.p
model = args.model
snapshot = args.snapshot
pspnet_prediction = args.pspnet_prediction

# Config
config = utils.get_config(project)
config["pspnet_prediction"] = pspnet_prediction

# Set up network
datasource = DataSource(config, random=False)
refine_net = RefineNet(datasource, model, snapshot)

root_result = build_output_dirname(project, snapshot)
print "Outputting to ", root_result
root_allprob = os.path.join(root_result, 'all_prob')

im_list = datasource.im_list
for im in im_list:
    print im

    fn_allprob = os.path.join(root_allprob, im.replace('.jpg', '.h5'))
    if os.path.exists(fn_allprob):
        print "Already done."
        continue
    # make paths if not exist
    if not os.path.exists(os.path.dirname(fn_allprob)):
        os.makedirs(os.path.dirname(fn_allprob))

    ap = refine_net.process(im)

    # write to file
    with h5py.File(fn_allprob, 'w') as f:
        f.create_dataset('allprob', data=ap)
