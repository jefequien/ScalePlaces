import os
import argparse
import random
import matplotlib.pyplot as plt
import numpy as np

from scipy import misc
import cv2
from ftdetect import features


#10,1000
def disp(img, name):
    # show the output of SLIC
    fig = plt.figure(name)
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(img)
    plt.axis("off")

parser = argparse.ArgumentParser()
parser.add_argument("-p", help="Project name")
parser.add_argument("-i", help="Image name")
args = parser.parse_args()

path = args.i
img = misc.imread(args.i)
disp(img, "img")

# Blur image
kernel = (5, 5)
blur = cv2.GaussianBlur(img, kernel, 0)

# Canny transform to get edges
median = np.median(img)
low_threshold = median * 0.3
high_threshold = median * 1.3
# low_threshold = 10
print low_threshold, high_threshold
edges = cv2.Canny(blur, low_threshold, high_threshold)
misc.imsave("edges.png", edges)
print edges.shape, edges.dtype

kernel = np.ones((3,3), np.uint8)
closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)
contours = [c for c,h in zip(contours, hierarchy[0]) if cv2.contourArea(c) > 100]

edges = np.zeros(img.shape)
cv2.drawContours(img, contours, -1, (255,0,0), 1)
misc.imsave("edges_again.png", img)


