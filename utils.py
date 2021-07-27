from PIL import Image, ImageOps
import matplotlib
import matplotlib.pyplot as plt
import cv2
import os
import imageio as io
import numpy as np

def display_image(img):
    plt.imshow(img)
    plt.show()

def save_numpy_image_INT(img,path):
    im = Image.fromarray(img)
    im.save(path)

def save_numpy_image_FLOAT(img,path):
    matplotlib.image.imsave(path, img)

def save_numpy_image_imageio(img,path):
    io.imwrite(path,img)

def remove_alpha_channel(img):
    if img.shape[2] == 4:
        img = img[:,:,:3]
    return img

def image_resizer_pil(img,resize_len):
    img = img.resize((resize_len, resize_len), Image.ANTIALIAS)
    return img

def image_resizer_cv2(img,resize_len):
    img = cv2.resize(img, (resize_len, resize_len))
    return img

def convert_rgb_greyscale(img):
    img = ImageOps.grayscale(img)
    return img

def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def compute_dice_score(x,y):
    k=1
    return np.sum(x[y==k])*2.0 / (np.sum(x) + np.sum(y))