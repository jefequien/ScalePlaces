import os
import numpy as np
from scipy import misc

from pspnet import PSPNet
import utils_run as utils

pspnet = PSPNet(DEVICE=2)
pspnet.get_network_architecture()

CONFIG = utils.get_data_config("ade20k")
im_list = utils.open_im_list("ade20k")
root_images = CONFIG["images"]

im_list = im_list[:100]
for im in im_list:
    print im
    image = misc.imread(os.path.join(root_images, im))
    probs = pspnet.process(image)

    cm_path = os.path.join("tmp/", im.replace('.jpg','.png'))
    if not os.path.exists(os.path.dirname(cm_path)):
        os.makedirs(os.path.dirname(cm_path))
        
    cm = np.argmax(probs, axis=0) + 1
    misc.imsave(cm_path, cm.astype('uint8'))


