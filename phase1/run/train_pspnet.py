import os
import random
import argparse
import sys

CAFFE_ROOT = '/data/vision/torralba/segmentation/places/PSPNet/'
sys.path.insert(0, os.path.join(CAFFE_ROOT, 'python'))
import caffe

ORI_WEIGHTS = '/data/vision/torralba/segmentation/places/PSPNet/evaluation/model/pspnet50_ADE20K.caffemodel'
SNAPSHOTS = '/data/vision/oliva/scenedataset/scaleplaces/ScalePlaces/phase1/run/snapshots/'

def get_model(solver_path):
    models = ["softmax", "sigmoid", "broden"]
    for model in models:
        prototxt = os.path.basename(solver_path)
        if model in prototxt:
            return model
    raise Exception('Model not found')

def get_latest_snapshot(snapshot_dir):
    latest_i = 0
    for fname in os.path.listdir():
        split = os.path.splitext(fname)
        if split[1] == '.solverstate':
            i = split[0].split('_')[2]
            i = int(i)
            if i > latest_i:
                latest_i = i
    fn_template = 'snapshot_iter_{}.solverstate'
    if i != 0:
        return os.path.join(snapshot_dir, fn_template.format(i))
    else:
        raise Exception('Snapshot not found')


parser = argparse.ArgumentParser()
parser.add_argument('--id', default=0,type=int)
parser.add_argument('--solver',required=True)
parser.add_argument('--resume', action='store_true', default=False)
args = parser.parse_args()

DEVICE = args.id
solver_path = args.solver
MODEL = get_model(solver_path)

caffe.set_mode_gpu()
caffe.set_device(DEVICE)
SEED = 3
random.seed(SEED)

solver = caffe.get_solver(solver_path)
if args.resume:
    snapshot_dir = os.path.join(SNAPSHOTS, MODEL)
    latest_snapshot = get_latest_snapshot(snapshot_dir)
    print "Resuming from latest snapshot: ", latest_snapshot
    solver.restore(latest_snapshot)
else:
    print "Finetuning for original weights."
    solver.net.copy_from(WEIGHTS)
solver.solve()
