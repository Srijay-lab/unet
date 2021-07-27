from model import *
from data import *
import os
import glob
from PIL import Image, ImageOps
import numpy as np
from utils import *
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

TRAIN_FOLDER = "F:/Datasets/CRAG_LabServer/SAFRON/c2/train/gland_segmentation"
TEST_FOLDER = "F:/Datasets/CRAG_LabServer/SAFRON/c2/unet_exp2_setup3_test/gland_segmentation"
TEST_RESULTS = "F:/Datasets/CRAG_LabServer/SAFRON/c2/unet_exp2_setup3_test/gland_segmentation/results/benign_traditional_augment_set"

IMAGE_SIZE = 256
mode = "train"
epochs = 50
batch_size = 2
model_file = 'safron/benign/crag_c2_traditional_augment_set.hdf5'
augment = True
gland_mask_threshold = 10


def read_image_and_mask(image_path,mask_path):
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

    # print(mask)
    # print(np.unique(mask))

    # scale mask values to 0 and 255 only
    mask[mask < gland_mask_threshold] = 0
    mask[mask >= gland_mask_threshold] = 255

    # normalize values
    image = image / 255.0
    mask = mask / 255.0

    # save_numpy_image_imageio(image, "image.png")
    # save_numpy_image_imageio(mask, "mask.png")

    if(augment):
        mask = np.expand_dims(mask, axis=2)

    return image,mask


def load_train_data():
    train_images_folder = os.path.join(TRAIN_FOLDER,"images/imgs")
    train_masks_folder = os.path.join(TRAIN_FOLDER,"masks/imgs")

    images = []
    masks = []

    mask_paths = glob.glob(os.path.join(train_masks_folder,"*.png"))

    for mask_path in mask_paths:
        imname = os.path.split(mask_path)[1]
        image_path = os.path.join(train_images_folder,imname)
        image,mask = read_image_and_mask(image_path,mask_path)
        images.append(image)
        masks.append(mask)

    images = np.array(images)
    masks = np.array(masks)

    return images,masks


def load_test_data():
    test_images_folder = os.path.join(TEST_FOLDER, "images")
    test_masks_folder = os.path.join(TEST_FOLDER, "masks")

    images = []
    masks = []
    image_names = []

    mask_paths = glob.glob(os.path.join(test_masks_folder, "*.png"))

    for mask_path in mask_paths:
        imname = os.path.split(mask_path)[1]
        image_names.append(imname)
        image_path = os.path.join(test_images_folder, imname)
        image, mask = read_image_and_mask(image_path, mask_path)
        images.append(image)
        masks.append(mask)

    images = np.array(images)
    masks = np.array(masks)

    return image_names, images, masks


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


def combine_generator(gen1, gen2):
    while True:
        yield(gen1.next(), gen2.next())

model = unet(augment=augment)

if(mode=="train"):

    model_checkpoint = ModelCheckpoint(model_file, monitor='loss',verbose=1, save_best_only=True)
    images,masks = load_train_data()
    data_len = len(masks)

    if(augment):

        seed = 1

        images_folder = os.path.join(TRAIN_FOLDER, "images")
        masks_folder = os.path.join(TRAIN_FOLDER, "masks")

        image_datagen = ImageDataGenerator()

        mask_datagen = ImageDataGenerator()

        # image_datagen.fit(images, augment=True, seed=seed)
        # mask_datagen.fit(masks, augment=True, seed=seed)

        image_generator = image_datagen.flow_from_directory(
            images_folder,
            class_mode=None,
            batch_size=batch_size,
            seed=seed)

        mask_generator = mask_datagen.flow_from_directory(
            masks_folder,
            class_mode=None,
            color_mode='grayscale',
            batch_size=batch_size,
            seed=seed)

        #train_gen = (pair for pair in zip(image_generator, mask_generator))
        train_gen = combine_generator(image_generator,mask_generator)

        # for i in range(9):
        #     # define subplot
        #     plt.subplot(330 + 1 + i)
        #
        #     # generate batch of images
        #     batch = next(train_gen)
        #     # convert to unsigned integers for viewing
        #     # image = batch[0][0].astype('uint8')
        #     # plt.imshow(image)
        #     # mask = batch[0][1].astype('uint8')
        #     # plt.imshow(mask)
        #     save_numpy_image_imageio(batch[0][0], "check/image_"+str(i)+".png")
        #     save_numpy_image_imageio(batch[1][0], "check/mask_"+str(i)+".png")
        # plt.show()
        # exit(0)

        model.fit_generator(train_gen,
                            steps_per_epoch=data_len/batch_size,
                            epochs=epochs,
                            callbacks=[model_checkpoint])
    else:
        model.fit(x=images,
                  y=masks,
                  batch_size=batch_size,
                  epochs=epochs,
                  callbacks=[model_checkpoint])
else:
    mkdir(TEST_RESULTS)

    model = load_model(model_file)
    print("Model loaded")

    image_names, gt_images, gt_masks = load_test_data()
    print("Data loaded")

    pred_masks = model.predict(gt_images,verbose=1)
    pred_masks = [binarize_segmentation_outputs(x) for x in pred_masks]

    dice1 = save_results(image_names, gt_masks,pred_masks,TEST_RESULTS,"gt_masks","pred_masks")

    print("dice1 :",dice1)
