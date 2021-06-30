from PIL import Image, ImageOps
import matplotlib
import matplotlib.pyplot as plt

def save_numpy_image_INT(img,path):
    im = Image.fromarray(img)
    im.save(path)

def save_numpy_image_FLOAT(img,path):
    matplotlib.image.imsave(path, img)

def remove_alpha_channel(img):
    if img.shape[2] == 4:
        img = img[:,:,:3]
    return img

def image_resizer_pil(img,resize_len):
    img = img.resize((resize_len, resize_len), Image.ANTIALIAS)
    return img

def convert_rgb_greyscale(img):
    img = ImageOps.grayscale(img)
    return img