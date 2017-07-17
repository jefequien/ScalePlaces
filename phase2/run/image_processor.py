import random
import numpy as np
from scipy import misc,ndimage

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
        canny = self.datasource.get_canny(idx)
        additional_features = [canny]

        data, label = self.build_top(ap, gt, additional_features=additional_features, n=n)
        return data,label

    def build_top(self, ap, gt, additional_features=[], n=None):
        '''
        Returns
        data nxcxhxw
        label nxhxw
        '''
        slices = self.get_slices(ap)
        random.shuffle(slices)
        if n is not None:
            while len(slices) < n:
                slices = np.concatenate([slices, slices], axis=0)
                slices = slices[:n]

        datas = []
        labels = []
        for i in slices:
            data,label = self.build_top_i(ap[i],gt[i],additional_features=additional_features)
            datas.append(data)
            labels.append(label)
        data = np.stack(datas)
        label = np.stack(labels)
        return data, label

    def build_top_i(self,img,gt,additional_features=[]):
        '''
        Builds top for a single slice.
        Returns:
        data cxhxw
        label hxw
        '''
        # Stack along c dimension
        features = [img]
        features += additional_features
        if len(features) > 1:
            img = np.concatenate(features, axis=0)

        # Rescale
        s = 473
        _,h,w = img.shape
        data = ndimage.zoom(img, (1.,1.*s/h,1.*s/w), order=1, prefilter=False, mode='constant')
        gt = np.squeeze(gt) 
        label = misc.imresize(gt,(s,s), interp='nearest') 
        return data, label

    def get_slices(self, ap):
        max_activation = [np.max(s) for s in ap]
        slices = np.argwhere(max_activation > self.threshold)
        return slices
