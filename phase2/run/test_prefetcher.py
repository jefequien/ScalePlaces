import os
from scipy import misc

import utils_run as utils
from data_source import DataSource
from prefetcher import PreFetcher
from prefetcher import *

def save(data,label):
    print data.shape, label.shape
    n = data.shape[0]
    for i in xrange(n):
        print i
        fn = "tmp/{}.jpg".format(i)
        misc.imsave(fn, data[i])
    fn = "tmp/label.jpg"
    misc.imsave(fn, label)
    print "saved"

project = 'ade20k'
config = utils.get_config(project)
config["canny"] = "canny"
batch_size = 5

data_source = DataSource(config, random=True)
prefetcher = PreFetcher(data_source, batch_size=batch_size, ahead=1)

for i in xrange(1):
    d = (data_source, i, batch_size)
    data, label = build_batch(d)
    #data, label = prefetcher.fetch_batch()
    save(data[0], label[0])
