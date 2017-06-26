

from pspnet import PSPNet
from data_handler import DataHandler


project = "ade20k"
im_list = "ade20k_training.txt"

# data_handler = DataHandler(project, im_list)
# pspnet = PSPNet(DEVICE=0)

solver = caffe.get_solver('solver_FCN.prototxt')


