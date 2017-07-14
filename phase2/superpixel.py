import os
import argparse
import matplotlib.pyplot as plt

from scipy import misc
from skimage.segmentation import slic, mark_boundaries
#10,1000

parser = argparse.ArgumentParser()
parser.add_argument("-p", help="Project name")
parser.add_argument("-i", help="Image name")
args = parser.parse_args()

path = args.i
img = misc.imread(args.i)


fig = plt.figure("")
ax = fig.add_subplot(1, 1, 1)
ax.imshow(img)
plt.axis("off")


for compactness in [10]:
    # apply SLIC and extract (approximately) the supplied number
    # of segments
    print compactness
    segments = slic(img, n_segments=1000, compactness = compactness, sigma = 1)
    print segments
 
    # show the output of SLIC
    fig = plt.figure("Superpixels -- {} compactness".format(compactness))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(mark_boundaries(img, segments))
    plt.axis("off")
 
# show the plots
plt.show()