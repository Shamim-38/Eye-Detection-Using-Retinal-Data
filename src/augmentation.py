""" this file contains several augmentation methods using skimage,scipy and PIL library """
"""
Notes:
1. augmentation must be done on training set not validation set
2. augmentation here are done on preprecessed images
3. augmenated images are saved on seperate folder
"""

# python modules
import os, random
import numpy as np 
import skimage.transform as tf
from scipy import ndarray, ndimage
from skimage import io, util, exposure


# project moduels
from .. import config

# augmentations method which we won't use in DRD
"""
def sigmoid_correction(img):
    return exposure.adjust_sigmoid(img)

def random_noise(img: ndarray):
    # add random noise to the image
    return util.random_noise(img, mode='gaussian')


def blur(img: ndarray):
    return ndimage.gaussian_filter(img, sigma = 2)
"""



# augmentation operations 
def deformation(image):
    random_shear_angl = np.random.random() * np.pi/6 - np.pi/12
    random_rot_angl = np.random.random() * np.pi/7 - np.pi/12 - random_shear_angl
    random_x_scale = np.random.random() * .4 + .8
    random_y_scale = np.random.random() * .4 + .8
    random_x_trans = np.random.random() * image.shape[0] / 4 - image.shape[0] / 8
    random_y_trans = np.random.random() * image.shape[1] / 4 - image.shape[1] / 8

    dx = image.shape[0]/2. \
            - random_x_scale * image.shape[0]/2 * np.cos(random_rot_angl)\
            + random_y_scale * image.shape[1]/2 * np.sin(random_rot_angl + random_shear_angl)

    dy = image.shape[1]/2. \
            - random_x_scale * image.shape[0]/2 * np.sin(random_rot_angl)\
            - random_y_scale * image.shape[1]/2 * np.cos(random_rot_angl + random_shear_angl)

    trans_mat = tf.AffineTransform(rotation = random_rot_angl,
                                translation=(dx + random_x_trans,
                                                dy + random_y_trans),
                                shear = random_shear_angl,
                                scale = (random_x_scale,random_y_scale))


    return tf.warp(image, trans_mat.inverse, 
                        output_shape = image.shape, mode='constant', cval=0.0) 



def image_deformation(image):
    tform = tf.SimilarityTransform(scale=1, rotation = np.random.random() * np.pi/12,
                               translation=(0, .1))
    
    return tf.warp(image, tform, mode='constant', cval=0.0)



def random_crop(img: ndarray):
    assert img.shape[2] == 3
    height, width = img.shape[0], img.shape[1]

    if (width > 700 and height > 700):
        # crop dimension
        c = 100
        dy, dx = (height - 2*c), (width - 2*c)

        # cropping location
        x = np.random.randint(0, c + 1)
        y = np.random.randint(0, c + 1)

        return img[y:(y+dy), x:(x+dx), :]

    else:
        return img



def random_rotation_neg(img: ndarray):
    # pick a random degree of rotation between 25% on the left and 25% on the right
    random_degree = random.uniform(-180, 0)
    return tf.rotate(img, random_degree)


def random_rotation_pos(img: ndarray):
    # pick a random degree of rotation between 25% on the left and 25% on the right
    random_degree = random.uniform(1, 180)
    return tf.rotate(img, random_degree)


def vertical_flip(img: ndarray):
    # horizontal flip doesn't need skimage, it's easy as flipping the image array of pixels !
    return img[::-1, :]
    

def shear(img: ndarray):
    afine_tf = tf.AffineTransform(shear = 0.2)
    return tf.warp(img, inverse_map = afine_tf, 
            mode='constant', cval=0.0)


def logarithmic_correction(img):
    return exposure.adjust_log(img)


def rescale_intensity(img):
    v_min, v_max = np.percentile(img, (0.2, 99.8))
    better_contrast = exposure.rescale_intensity(img, in_range=(v_min, v_max))
    return better_contrast



# make these operations for hozinotal image flipping
def horizontal_image_deformation(img: ndarray):
    return image_deformation(img[:, ::-1])


def horizontal_deformation(img: ndarray):
    return deformation(img[:, ::-1])


def horizontal_shear(img: ndarray):
    return shear(img[:, ::-1])


def horizontal_flip(img: ndarray):
    return img[:, ::-1]



# for augmentation
def do_augmentation(class_id, class_aug_opearations):
    print("start augmenting ...")
    
    # setup input output directory 
    input_dir = os.path.join(config.data_path(), "train", class_id)
    output_dir = input_dir
    os.makedirs(output_dir, exist_ok = True)

    all_preprocessed_images = os.listdir(input_dir)
    ls_image_path = [os.path.join(input_dir, path) for path in all_preprocessed_images]

    # do augmentation for given operation
    # for each preprocess image
    for path in ls_image_path:
        img = io.imread(path)

        # for each augmentation
        for key in class_aug_opearations.keys():
            trans_img = class_aug_opearations[key](img)
            
            # write image to the disk
            new_img_file = "aug_" + key + "_" + "_".join((path.split("/")[-1]).split("_")[1:])
            new_image_path = os.path.join(output_dir, new_img_file)
            io.imsave(new_image_path, trans_img)