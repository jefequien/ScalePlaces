import numpy as np

from image_processor import ImageProcessor

class BatchBuilder:
    def __init__(self, datasource, batch_size=5):
        self.datasource = datasource
        self.image_processor = ImageProcessor(self.datasource)

        self.batch_size = batch_size
        self.batch = (None, None)

    def build_batch(self):
        data, label = self.batch
        while data is None or data.shape[0] <= self.batch_size:
            idx = self.datasource.next_idx()
            new_data,new_label = self.image_processor.process(idx)

            data = np.concatenate((data, new_data), axis=0)
            label = np.concatenate((label, new_data), axis=0)

        batch_data = data[:self.batch_size, ...]
        batch_label = label[:self.batch_size, ...]
        r_data = data[self.batch_size, ...]
        r_label = label[self.batch_size, ...]
        
        self.batch = (r_data, r_label)
        return batch_data, batch_label
