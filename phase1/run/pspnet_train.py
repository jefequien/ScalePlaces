import os
import argparse

CAFFE_ROOT = '/data/vision/torralba/segmentation/places/PSPNet/'
sys.path.insert(0, os.path.join(CAFFE_ROOT, 'python'))
import caffe

WEIGHTS = '/data/vision/torralba/segmentation/places/PSPNet/evaluation/model/pspnet50_ADE20K.caffemodel'

parser = argparse.ArgumentParser()
parser.add_argument('--id', default=0,type=int)
args = parser.parse_args()

DEVICE = args.id

caffe.set_mode_gpu()
caffe.set_device(DEVICE)

SEED = 3
random.seed(SEED)

solver = caffe.get_solver('models/solver_pspnet_with_data_layer.prototxt')
solver.net.copy_from(WEIGHTS)
solver.solve()