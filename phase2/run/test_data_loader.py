import os
from scipy import misc

import utils_run as utils
from data_source import DataSource
from prefetcher import PreFetcher
from prefetcher import *

def save(data,label):
    output = "tmp/"
    for i in data:
        fn = "{}.png".format(i)
        misc.imsave(os.path.join(output, fn), data[i])
    misc.imsave(os.path.join(output, fn), label)

project = 'ade20k'
config = utils.get_config(project)
batch_size = 5

data_source = DataSource(config, random=True)
prefetcher = PreFetcher(data_source, batch_size=batch_size, ahead=12)

for i in xrange(0):
    #d = (data_source, i, batch_size)
    #data, label = build_batch(d)
    data, label = prefetcher.fetch_batch()
    print data.shape, label.shape

    save(data[0], label[0])
