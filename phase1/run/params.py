
original_weights = '/data/vision/torralba/segmentation/places/PSPNet/evaluation/model/pspnet50_ADE20K.caffemodel'

model_softmax = 'models/pspnet50_ADE20K_473.prototxt'
model_sigmoid = 'models/pspnet_sigmoid.prototxt'


snapshots_dir = '/data/vision/oliva/scenedataset/scaleplaces/ScalePlaces/phase1/run/snapshots/'
softmax = snapshots_dir + "softmax/snapshot_iter_{}.caffemodel".format(300000)
sigmoid = snapshots_dir + "sigmoid/snapshot_iter_{}.caffemodel".format(26000)
sigmoid_slower = snapshots_dir + "sigmoid_slower/snapshot_iter_{}.caffemodel".format(22000)


WEIGHTS = softmax
MODEL = model_sigmoid