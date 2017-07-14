from __future__ import division
from gaussian_filter import gaussian
from gradient import gradient
from nonmax_suppression import maximum
from double_thresholding import thresholding
from numpy import array, zeros
import numpy as np
from PIL import Image
from matplotlib.pyplot import imshow, show, subplot, gray, title, axis

from scipy import misc
import time

class tracking:
    def __init__(self, tr):
        self.im = tr[0]
        strongs = tr[1]

        self.vis = zeros(im.shape, bool)
        self.dx = [1, 0, -1,  0, -1, -1, 1,  1]
        self.dy = [0, 1,  0, -1,  1, -1, 1, -1]
        for s in strongs:
            if not self.vis[s]:
                self.dfs(s)
        for i in xrange(self.im.shape[0]):
            for j in xrange(self.im.shape[1]):
                self.im[i, j] = 1.0 if self.vis[i, j] else 0.0

    def dfs(self, origin):
        q = [origin]
        while len(q) > 0:
            s = q.pop()
            self.vis[s] = True
            self.im[s] = 1
            for k in xrange(len(self.dx)):
                for c in xrange(1, 16):
                    nx, ny = s[0] + c * self.dx[k], s[1] + c * self.dy[k]
                    if self.exists(nx, ny) and (self.im[nx, ny] >= 0.5) and (not self.vis[nx, ny]):
                        q.append((nx, ny))
        pass

    def exists(self, x, y):
        return x >= 0 and x < self.im.shape[0] and y >= 0 and y < self.im.shape[1]

def run(img):
    gim = gaussian(img)
    grim, gphase = gradient(gim)
    gmax = maximum(grim, gphase)
    # gmax = (gmax/np.max(gmax))*255
    # return gmax.astype('uint8')
    return gmax


if __name__ == '__main__':
    from sys import argv
    im = misc.imread(argv[1])
    t = time.time()
    gmax = run(im)
    print gmax.shape, gmax.dtype
    print np.min(gmax), np.max(gmax)

    # misc.imsave("woo1.png",gmax.astype('uint8'))
    print time.time() - t
    thres = thresholding(gmax)
    edge = tracking(thres)

    gray()
    subplot(1, 2, 1)
    imshow()
    axis('off')
    title('double')

    subplot(1, 2, 2)
    imshow(edge.im)
    axis('off')
    title('Edges')

    show()
