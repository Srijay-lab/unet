from model import *
from data import *
import os
import glob
from PIL import Image, ImageOps
import numpy as np
from utils import *

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

TRAIN_FOLDER = "D:/warwick/datasets/digestpath/train_data/benign"
IMAGE_SIZE = 256

train_images_folder = os.path.join(TRAIN_FOLDER,"images")
train_masks_folder = os.path.join(TRAIN_FOLDER,"masks")

images = []
masks = []

mask_paths = glob.glob(os.path.join(train_masks_folder,"*.png"))

for mask_path in mask_paths:
    imname = os.path.split(mask_path)[1]
    image_path = os.path.join(train_images_folder,imname)

    image_pil = Image.open(image_path)
    mask_pil = Image.open(mask_path)
    imsize = mask_pil.size[0]

    #Resize
    if(imsize != IMAGE_SIZE):
        image_pil = image_resizer_pil(image_pil,IMAGE_SIZE)
        mask_pil = image_resizer_pil(mask_pil,IMAGE_SIZE)

    #convert mask to greyscale
    mask_pil = convert_rgb_greyscale(mask_pil)

    #Numpy
    image = np.array(image_pil)
    mask = np.array(mask_pil)

    #Remove alpha channel of image
    image = remove_alpha_channel(image)

    # scale mask values to 0 and 255 only
    mask[mask < 100] = 0
    mask[mask >= 100] = 255

    save_numpy_image_FLOAT(image,"image.png")
    save_numpy_image_FLOAT(mask,"mask.png")

    #normalize values
    image = image/255.0
    mask = mask/255.0

    images.append(image)

    mask = np.expand_dims(mask, axis=2)
    masks.append(mask)


images = np.array(images)
masks = np.array(masks)

model = unet()
model_checkpoint = ModelCheckpoint('unet_membrane1.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit(x=images,
          y=masks,
          batch_size=2,
          epochs=30,
          callbacks=[model_checkpoint])

exit(0)

testGene = testGenerator("data/membrane/test")
results = model.predict_generator(testGene,30,verbose=1)
saveResult("data/membrane/test1",results)