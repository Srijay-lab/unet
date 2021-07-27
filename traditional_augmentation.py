import numpy as np
from skimage.transform import rotate

def horizontal_flip(img):
    return np.fliplr(img)

def vertical_flip(img):
    return np.flipud(img)

