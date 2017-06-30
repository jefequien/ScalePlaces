import sys
import h5py
import numpy as np

def get_datasets(fname):
    datasets = {}
    with h5py.File(fname, 'r') as f:
        for name in f['.']:
            output = f[name]
            datasets[name] = output[:]
    return datasets


fnames = sys.argv[1:]
keys = set()
datasets = []
for f in fnames:
    f_datasets = get_datasets(f)
    keys |= set(f_datasets.keys())
    datasets.append(f_datasets)

stacked = {}
for key in keys:
    arrays = []
    for dataset in datasets:
        if key in dataset:
            arrays.append(dataset[key])
    stacked[key] = np.concatenate(arrays, axis=0)
    
fname = "merged.h5"
with h5py.File(fname, 'w') as f:
    for key in stacked:
        print key, stacked[key].shape
        f.create_dataset(key, data=stacked[key])

