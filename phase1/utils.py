from scipy import misc
import matplotlib.pyplot as plt
import cv2
import numpy as np
import random

def get_categories():
    categories = {}
    with open("objectInfo150.txt", 'r') as f:
        for line in f.readlines():
            split = line.split()
            cat = split[0]
            if cat.isdigit():
                categories[int(cat)] = split[4].replace(',','')
        return categories



def apply_mask(image, mask):
    masked_image = np.copy(image)

    color = [255,0,0]
    random.shuffle(color)
    masked_image[mask] = np.maximum(masked_image[mask], color)

    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(masked_image, contours, -1, (0, 0, 0), 2)
    return masked_image


image_path = "/Users/hujh/Documents/UROP_Torralba/ADE_20K/images/ADE_train_00000037.jpg"
mask_path = "/Users/hujh/Documents/UROP_Torralba/ADE_20K/annotations/ADE_train_00000037.png"
category = 1
image = misc.imread(image_path)
mask = misc.imread(mask_path)

category_mask = (mask == category)
masked_image = apply_mask(image, category_mask)

plt.imshow(masked_image)
plt.show()