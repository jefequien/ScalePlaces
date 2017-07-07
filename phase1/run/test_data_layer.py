import os
import uuid
import caffe

from data_layer_image import DataLayer


def save(img):
    fname = "tmp/{}.jpg".format(uuid.uuid4().hex)
    misc.imsave(fname, img)
    return path

layer = DataLayer()

top = [None,None]
bottom = []
layer.setup(bottom,top)

layer.reshape(bottom,top)

save(layer.data)
save(layer.label)

