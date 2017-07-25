import os
import sys
import argparse
import numpy as np

from scipy import misc
import cv2

def make_canny(img):
    print "Making canny..."
    # Blur image
    kernel = (5, 5)
    blur = cv2.GaussianBlur(img, kernel, 0)

    median = np.median(img)
    low_threshold = median * 0.3
    high_threshold = median * 1.5
    edges = cv2.Canny(blur, low_threshold, high_threshold)
    return edges

if __name__=="__main__":
    im = sys.argv[1]
    print im
    img = misc.imread(im)
    # img = cv2.imread(im)
    edges = canny(img)

    # misc.imsave("edges_m.png", edges)
    cv2.imshow("", edges)
    cv2.waitKey(0)

