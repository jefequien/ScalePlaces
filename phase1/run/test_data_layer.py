import os
import uuid
import caffe
from scipy import misc

from data_layer_image import DataLayer


def save(img):
    fname = "tmp/{}.jpg".format(uuid.uuid4().hex)
    misc.imsave(fname, img)

layer = DataLayer()

top = [None,None]
bottom = []
layer.setup(bottom,top)

layer.reshape(bottom,top)

data = layer.data
data = data.transpose((1,2,0))
label = layer.label

print data.shape
print label.shape
save(data)
save(label)

