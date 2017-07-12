import argparse
import os
import uuid
import time
import cv2

from scipy import misc
import numpy as np

import utils_vis as utils

THRESHOLD = False
INDIV_SLICES = True

class ImageVisualizer:

    def __init__(self, project, special_config=None):
        self.images_dir = "tmp/images/"
        if not os.path.exists(self.images_dir):
            os.makedirs(self.images_dir)

        self.config = utils.get_config(project)
        if special_config is not None:
            self.config = special_config

    def visualize(self, im):
        paths = {}
        im_path = utils.get_file_path(im, self.config, ftype="im")
        paths["image"] = im_path

        cm, cm_path = self.get_category_mask(im)
        if cm is not None:
            cm_color, cm_color_path = self.add_color(cm)
            paths["category_mask"] = cm_color_path

        pm, pm_path = self.get_prob_mask(im)
        if pm is not None:
            paths["prob_mask"] = pm_path

        gt, gt_path = self.get_ground_truth(im)
        if gt is not None:
            gt_color, gt_color_path = self.add_color(gt)
            paths["ground_truth"] = gt_color_path

        diff = self.get_diff(cm, gt)
        if diff is not None:
            diff_color, diff_color_path = self.add_color(diff)
            paths["diff"] = diff_color_path

        ap, ap_path = self.get_all_prob(im)
        if THRESHOLD and cm is not None and ap is not None:
            thresholds = self.get_thresholds(ap, cm)
            thresholds_color, thresholds_color_path = self.add_color(thresholds)
            paths["thresholds"] = thresholds_color_path

        if INDIV_SLICES and ap is not None:
            indiv_slices = self.get_individual_slices(ap, 20)
            indiv_slices_path = self.save(indiv_slices)
            #indiv_slices_color, indiv_slices_color_path = self.add_color(indiv_slices)
            paths["indiv_slices"] = indiv_slices_path

        return paths

    def get_category_mask(self, im):
        try:
            path = utils.get_file_path(im, self.config, ftype="cm")
            img = utils.get_file(im, self.config, ftype="cm")
            return img, path
        except:
            print "No category mask", im
            return None, None

    def get_prob_mask(self, im):
        try:
            path = utils.get_file_path(im, self.config, ftype="pm")
            img = utils.get_file(im, self.config, ftype="pm")
            return img, path
        except:
            print "No prob mask", im
            return None, None

    def get_ground_truth(self, im):
        try:
            path = utils.get_file_path(im, self.config, ftype="gt")
            img = utils.get_file(im, self.config, ftype="gt")
            return img, path
        except:
            #print "No ground_truth", im
            return None, None

    def get_all_prob(self, im):
        try:
            path = utils.get_file_path(im, self.config, ftype="ap")
            img = utils.get_file(im, self.config, ftype="ap")
            return img, path
        except:
            print "No all prob", im
            return None, None

    def get_diff(self, cm, gt):
        if cm is None or gt is None:
            print "Cannot make diff"
            return None
        mask = gt - cm
        mask = np.invert(mask.astype(bool))
        diff = np.copy(gt)
        diff[mask] = 0
        return diff

    def get_thresholds(self, ap, cm):
        thresholds = np.linspace(0.5,1,11)
        all_imgs = []
        for threshold in thresholds:
            img = np.zeros(cm.shape)
            for i in xrange(150):
                c = i+1
                probs = ap[i,:,:]
                prob_mask = probs > threshold
                cm_mask = cm == c
                
                mask = np.logical_and(prob_mask, cm_mask)
                img[mask] = c
            all_imgs.append(img)
        return np.concatenate(all_imgs, axis=1)

    def get_individual_slices(self, ap, n):
        threshold = 0.5
        K,h,w = ap.shape

        ap = ap > threshold
        sums = [np.sum(slic) for slic in ap]
        top_slices = np.flip(np.argsort(sums), 0)

        labeled_slices = []
        for i in top_slices[:n]:
            c = i+1
            slic = ap[i,:,:]
            slic = slic > threshold

            labeled = self.label_img(slic, c)
            labeled_slices.append(labeled)
        output = np.concatenate(labeled_slices, axis=1)
        return output

    def label_img(self, img, c):
        if img.dtype == bool:
            img = img.astype(int)*255
        color = utils.to_color(c)
        if np.ndim(img) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img[:50,:100,:] = color
        cv2.putText(img, str(c), (0,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0))
        return img

    def add_color(self, img):
        if img is None:
            return None, None

        h,w = img.shape
        img_color = np.zeros((h,w,3))
        for i in xrange(1,151):
            img_color[img == i] = utils.to_color(i)
        path = self.save(img_color)
        return img_color, path

    def save(self, img):
        fname = "{}.jpg".format(uuid.uuid4().hex)
        path = os.path.join(self.images_dir, fname)
        misc.imsave(path, img)
        return path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", required=True, help="Project name")
    parser.add_argument("-i", help="Image name")
    args = parser.parse_args()

    project = args.p
    im = args.i
    if not im:
        im_list = utils.open_im_list(project)
        im = im_list[0]

    print project, im
    vis = ImageVisualizer(project)
    paths = vis.visualize(im)
    print paths

