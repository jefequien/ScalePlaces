import numpy as np


class BatchBuilder:
    def __init__(self, datasource, batch_size=4):
        self.datasource = datasource
        self.batch_size = batch_size

    def build_batch(self):
        datas = []
        labels = []
        for i in xrange(batch_size):
            idx = self.datasource.next_idx()
            img = self.datasource.get_image(idx)
            gt = self.datasource.get_ground_truth(idx)
            ap = self.datasource.get_all_prob(idx)
            canny = self.datasource.get_canny(idx)

            data, label = self.build_top(img,gt,ap,canny)
            datas.append(data)
            labels.append(label)
        data = np.concatenate(datas, axis=0)
        label = np.concatenate(labels, axis=0)
        return data, label

    def build_top(self,img,gt,ap,canny):
        threshold = 0.5

        max_activation = [np.max(s) for s in ap]
        slices = np.argwhere(max_activation > threshold)

        datas = []
        labels = []
        for i in slices:
            data, label = self.build_top
            datas.append



    def build_top_per_slice(self,img,gt,ap,canny,i=-1):
        '''
        Data: 2xhxw
        Label: 1xhxw
        '''
        h,w,_ = img.shape
        channels = [ap[s],canny]
        data = np.stack(channels)

        c = i+1
        label = gt == c
        label = label[np.newaxis,:,:]
        return label, gt


