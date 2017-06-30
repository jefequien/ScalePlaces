import sys
import h5py

def open_data(fname, dataset_name):
    with h5py.File(fname, 'r') as f:
        output = f[dataset_name]
        return output[:]
def get_datasets(fname):
    datasets = {}
    with h5py.File(fname, 'r') as f:
        for name in f['.']:
            output = f[name]
            datasets[name] = output[:]
    return datasets


f1 = sys.argv[1]
f2 = sys.argv[2]

f1_datasets = get_datasets(f1)
f2_datasets = get_datasets(f2)

print f1_datasets.keys()
print f2_datasets.keys()