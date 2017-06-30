import os
import numpy as np
from scipy import misc

from pspnet import PSPNet
import utils_run as utils

def add_color(img):
    h,w = img.shape
    img_color = np.zeros((h,w,3))
    for i in xrange(1,151):
        img_color[img == i] = utils.to_color(i)
    return img_color

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
    cm = cm.astype('uint8')
    
    color = add_color(cm)
    misc.imsave(cm_path, color)


