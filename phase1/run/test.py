import caffe


solver = caffe.get_solver('models/solver.prototxt')

for k,v in solver.net.blobs.items():
    print v.data.shape, k
    
solver.solve()
