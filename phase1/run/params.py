
original_weights = '/data/vision/torralba/segmentation/places/PSPNet/evaluation/model/pspnet50_ADE20K.caffemodel'

model_softmax = 'models/test_pspnet_softmax.prototxt'
model_sigmoid = 'models/test_pspnet_sigmoid.prototxt'


snapshots_dir = '/data/vision/oliva/scenedataset/scaleplaces/ScalePlaces/phase1/run/snapshots/'
softmax = snapshots_dir + "softmax/snapshot_iter_{}.caffemodel".format(300000)
sigmoid = snapshots_dir + "sigmoid/snapshot_iter_{}.caffemodel".format(26000)
sigmoid_slower = snapshots_dir + "sigmoid_slower/snapshot_iter_{}.caffemodel".format(22000)


WEIGHTS = sigmoid_slower
MODEL = model_sigmoid
