from model import *
from data import *
import os
import glob
from PIL import Image, ImageOps
import numpy as np
from utils import *
from tensorflow.keras.models import load_model

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# TRAIN_FOLDER = "D:/warwick/datasets/digestpath/train_data/benign"
# TEST_FOLDER = "D:/warwick/datasets/digestpath/train_data/benign/test_results"
# TEST_RESULTS = "D:/warwick/datasets/digestpath/train_data/benign/unet_test_results"

TRAIN_FOLDER = "F:/Datasets/DigestPath/scene_generation/onlybenign/old_split_exp10_v0/train_data/train"
TEST_FOLDER = "C:/Users/Srijay/Desktop/Projects/scene_graph_pathology/training_outputs/prev_experiments/test_10"
TEST_RESULTS = "C:/Users/Srijay/Desktop/Projects/Keras/unet/results/exp10-2"

IMAGE_SIZE = 256
mode = "test"
epochs = 50
model_file = 'unet-scene-exp10-2.hdf5'


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
    mask[mask < 40] = 0
    mask[mask >= 40] = 255

    # normalize values
    image = image / 255.0
    mask = mask / 255.0

    # save_numpy_image_imageio(image, "image.png")
    # save_numpy_image_imageio(mask, "mask.png")
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
    image_names = []

    for path in paths:
        imname = os.path.split(path)[1]
        if("gt_image" in path):
            image_names.append(imname.split(".")[0])
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

    return image_names,gt_scenegeneration_images,gt_scenegeneration_masks,pred_scenegeneration_images,pred_scenegeneration_masks


def save_results(image_names, gt_images,pred_images,folder,gt_name,pred_name):
    l = len(gt_images)
    dice_score = 0
    for i in range(0,l):
        save_numpy_image_imageio(gt_images[i],os.path.join(folder,image_names[i]+"_"+gt_name+".png"))
        save_numpy_image_imageio(pred_images[i],os.path.join(folder,image_names[i]+"_"+pred_name+".png"))
        dice_score += compute_dice_score(gt_images[i],pred_images[i])
    dice_score=dice_score/l*1.0
    print("-----------------------------------------Dice score between "+gt_name+" and "+pred_name + " is "+str(dice_score))
    return dice_score


def binarize_segmentation_outputs(x):
    x[x < 0.5] = 0
    x[x >= 0.5] = 1.0
    return x


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

    image_names, gt_scenegeneration_images, gt_scenegeneration_masks, pred_scenegeneration_images, pred_scenegeneration_masks = load_test_data_scenegeneration_results()
    print("Data loaded")

    gt_scene_mask_predictions = model.predict(gt_scenegeneration_images,verbose=1)
    pred_scene_mask_predictions = model.predict(pred_scenegeneration_images,verbose=1)

    gt_scene_mask_predictions = [binarize_segmentation_outputs(x) for x in gt_scene_mask_predictions]
    pred_scene_mask_predictions = [binarize_segmentation_outputs(x) for x in pred_scene_mask_predictions]

    dice1 = save_results(image_names, gt_scenegeneration_masks,gt_scene_mask_predictions,TEST_RESULTS,"gt_scene_gt_masks","gt_scene_pred_masks")
    dice2 = save_results(image_names, pred_scenegeneration_masks,pred_scene_mask_predictions,TEST_RESULTS,"pred_scene_gt_masks","pred_scene_pred_masks")

    print("dice1 :",dice1)
    print("dice2 :",dice2)