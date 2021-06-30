from model import *
from data import *
import os
import glob
from PIL import Image, ImageOps
import numpy as np
from utils import *
from tensorflow.keras.models import load_model

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

TRAIN_FOLDER = "D:/warwick/datasets/digestpath/train_data/benign"
TEST_FOLDER = "D:/warwick/datasets/digestpath/train_data/benign/test_results"
TEST_RESULTS = "D:/warwick/datasets/digestpath/train_data/benign/unet_test_results"
IMAGE_SIZE = 256
mode = "test"
epochs = 3
model_file = 'unet_membrane.hdf5'


def read_image_mask(image_path,mask_path):
    image_pil = Image.open(image_path)
    mask_pil = Image.open(mask_path)
    imsize = mask_pil.size[0]

    # convert mask to greyscale
    mask_pil = convert_rgb_greyscale(mask_pil)

    # Numpy
    image = np.array(image_pil)
    mask = np.array(mask_pil)

    # Resize
    if (imsize != IMAGE_SIZE):
        image = image_resizer_cv2(image, IMAGE_SIZE)
        mask = image_resizer_cv2(mask, IMAGE_SIZE)

    # Remove alpha channel of image
    image = remove_alpha_channel(image)

    # scale mask values to 0 and 255 only
    mask[mask < 100] = 0
    mask[mask >= 100] = 255

    # normalize values
    image = image / 255.0
    mask = mask / 255.0

    # save_numpy_image_FLOAT(image, "image.png")
    # save_numpy_image_FLOAT(mask, "mask.png")
    # exit(0)

    #mask = np.expand_dims(mask, axis=2)

    return image,mask


def load_train_digestpath_data():
    train_images_folder = os.path.join(TRAIN_FOLDER,"images")
    train_masks_folder = os.path.join(TRAIN_FOLDER,"masks")

    images = []
    masks = []

    mask_paths = glob.glob(os.path.join(train_masks_folder,"*.png"))

    for mask_path in mask_paths:
        imname = os.path.split(mask_path)[1]
        image_path = os.path.join(train_images_folder,imname)
        image,mask = read_image_mask(image_path,mask_path)
        images.append(image)
        masks.append(mask)


    images = np.array(images)
    masks = np.array(masks)

    return images,masks


def load_test_data_scenegeneration_results():

    paths = glob.glob(os.path.join(TEST_FOLDER, "*.png"))
    gt_scenegeneration_images = []
    gt_scenegeneration_masks = []
    pred_scenegeneration_images = []
    pred_scenegeneration_masks = []

    for path in paths:
        imname = os.path.split(path)[1]
        if("gt_image" in path):
            gt_image_path = path
            pred_image_path = os.path.join(TEST_FOLDER,imname.replace("gt_image","pred_image"))
            gt_mask_path = os.path.join(TEST_FOLDER,imname.replace("gt_image","gt_mask"))
            pred_mask_path = os.path.join(TEST_FOLDER,imname.replace("gt_image","pred_trimask"))
            gt_image,gt_mask = read_image_mask(gt_image_path,gt_mask_path)
            pred_image,pred_mask = read_image_mask(pred_image_path,pred_mask_path)

            gt_scenegeneration_images.append(gt_image)
            gt_scenegeneration_masks.append(gt_mask)
            pred_scenegeneration_images.append(pred_image)
            pred_scenegeneration_masks.append(pred_mask)

    gt_scenegeneration_images = np.array(gt_scenegeneration_images)
    gt_scenegeneration_masks = np.array(gt_scenegeneration_masks)
    pred_scenegeneration_images = np.array(pred_scenegeneration_images)
    pred_scenegeneration_masks = np.array(pred_scenegeneration_masks)

    return gt_scenegeneration_images,gt_scenegeneration_masks,pred_scenegeneration_images,pred_scenegeneration_masks


def save_results(gt_images,pred_images,folder,gt_name,pred_name):
    l = len(gt_images)
    for i in range(0,l):
        save_numpy_image_FLOAT(gt_images[i],os.path.join(folder,str(i)+"_"+gt_name+".png"))
        save_numpy_image_FLOAT(pred_images[i],os.path.join(folder,str(i)+"_"+pred_name+".png"))


model = unet()

if(mode=="train"):
    model_checkpoint = ModelCheckpoint(model_file, monitor='loss',verbose=1, save_best_only=True)
    images,masks = load_train_digestpath_data()
    model.fit(x=images,
              y=masks,
              batch_size=2,
              epochs=epochs,
              callbacks=[model_checkpoint])
else:
    mkdir(TEST_RESULTS)

    model = load_model(model_file)
    print("Model loaded")

    gt_scenegeneration_images, gt_scenegeneration_masks, pred_scenegeneration_images, pred_scenegeneration_masks = load_test_data_scenegeneration_results()
    print("Data loaded")

    gt_scene_mask_predictions = model.predict(gt_scenegeneration_images,verbose=1)
    pred_scene_mask_predictions = model.predict(pred_scenegeneration_images,verbose=1)

    save_results(gt_scenegeneration_masks,gt_scene_mask_predictions,TEST_RESULTS,"gt_scene_gt_masks","gt_scene_pred_masks")
    save_results(pred_scenegeneration_masks,pred_scene_mask_predictions,TEST_RESULTS,"pred_scene_gt_masks","pred_scene_pred_masks")