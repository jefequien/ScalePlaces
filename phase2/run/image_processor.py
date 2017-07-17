import numpy as np

class ImageProcessor:
    def __init__(self, datasource):
        self.datasource = datasource
        self.threshold = 0.5

    def process(self, idx, n=None):
        ap = self.datasource.get_all_prob(idx)
        gt = self.datasource.get_ground_truth(idx)
        # One hot encode gt
        NUM_CLASS = 150
        gt = (np.arange(NUM_CLASS) == gt[:,:,None] - 1)
        gt = gt.transpose((2,0,1))

        # Load additional features
        img = self.datasource.get_image(idx)
        # canny = self.datasource.get_canny(idx)
        additional_features = [img]

        data, label = self.build_top(ap, gt, additional_features=additional_features, n=n)
        return data,label

    def build_top(self, ap, gt, additional_features=None, n=None):
        '''
        Returns
        data nxcxhxw
        label nxhxw
        '''
        slices = self.get_slices(ap)
        if n is not None:
            while len(slices) < n:
                slices = slices + slices
                slices = slices[:n]

        datas = []
        labels = []
        for i in slices:
            data,label = self.build_top(ap[s],gt[s],additional_features)
            datas.append(data)
            labels.append(label)
        data = np.stack(datas)
        label = np.stack(labels)
        return data, label

    def build_top_i(self,img,gt,additional_features=None):
        '''
        Builds top for a single slice.
        Returns:
        data cxhxw
        label hxw
        '''
        # Stack along c dimension
        if additional_features is not None:
            additional_features.append(img)
            img = np.concatenate(additional_features, axis=0)

        # Crop and scale
        # box = self.random_crop()
        s = 473
        data = data[:,:s,:s]
        label = label[:,:s,:s]
        # data = self.crop(img, box)
        # label = self.crop(label, box)
        return data, label

    def get_slices(self, ap):
        max_activation = [np.max(s) for s in ap]
        slices = np.argwhere(max_activation > self.threshold)
        return slices

def prep(features, gt):
    '''
    Crop and scale box to 473x473
    Returns
    data:  len(features)xhxw
    label: 1xhxw
    '''
    pass
