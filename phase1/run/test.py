import caffe


solver = caffe.get_solver('models/solver.prototxt')
#solver.solve()

for k,v in solver.net.blobs.items():
    print v.data.shape, k
