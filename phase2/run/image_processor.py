import random
import numpy as np
from scipy import misc,ndimage

class ImageProcessor:
    def __init__(self, datasource):
        self.datasource = datasource
        self.threshold = 0.5*255 # ap has been scaled by 255

    def build_data(self, idx):
        ap = self.datasource.get_all_prob(idx)
        slices = self.get_slices(ap)

        # Load additional features
        img = self.datasource.get_image(idx)
        canny = self.datasource.get_canny(idx)
        additional_features = [canny]

        data = self.build_top(ap, slices, additional_features=additional_features)
        return data,label

    def build_data_and_label(self, idx, batch_size=None):
        ap = self.datasource.get_all_prob(idx)
        # One hot encoded gt
        gt = self.datasource.get_ground_truth(idx)
        NUM_CLASS = 150
        gt = (np.arange(NUM_CLASS) == gt[:,:,None] - 1)
        gt = gt.transpose((2,0,1))

        slices = self.get_slices(ap)
        # Make slices = batch size
        if batch_size is not None:
            random.shuffle(slices)
            while len(slices) < batch_size:
                slices = np.concatenate([slices, slices], axis=0)
            slices = slices[:batch_size]

        # Load additional features
        img = self.datasource.get_image(idx)
        canny = self.datasource.get_canny(idx)
        additional_features = [canny]
        
        data = self.build_top(ap, slices, additional_features=additional_features)
        label = self.build_top(gt, slices)
        label = np.squeeze(label)
        return data,label

    def build_top(self, a, slices, additional_features=[]):
        '''
        Returns top: nxcxhxw
        '''
        top_slices = [self.build_top_i(a[i], additional_features=additional_features) for i in slices]
        data = np.stack(top_slices)
        return data

    def build_top_i(self,s,additional_features=[]):
        '''
        Builds top for a single slice. Returns slice: cxhxw
        '''
        # Stack along c dimension
        features = [s] + additional_features
        features = [a[np.newaxis,:,:] for a in features if np.ndim(a) != 3]
        data = None
        if len(features) == 1:
            data = features[0]
        else:
            data = np.concatenate(features, axis=0)
        
        # Rescale
        s = 473
        _,h,w = data.shape
        data = ndimage.zoom(data, (1.,1.*s/h,1.*s/w), order=1, prefilter=False, mode='constant')
        
        # data = ndimage.zoom(data, (1.,1.*s/h,1.*s/w), order=1, prefilter=False, mode='constant')
        # label = misc.imresize(gt,(s,s), interp='nearest') 
        # label = label == 255
        return data

    def get_slices(self, ap):
        max_activation = np.array([np.max(s) for s in ap])
        slices = np.argwhere(max_activation > self.threshold)
        return slices.flatten()
