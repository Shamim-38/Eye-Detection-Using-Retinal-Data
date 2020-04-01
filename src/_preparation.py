"""this file prepare train and validation data for diabetic retinopathy detection"""

# python modules
import numpy as np 
import os
from PIL import Image
import PIL
from skimage import io

# project moduels
from .. import config

# path variables and constant
left_img_dir = os.path.join(config.data_path(), "left")
right_img_dir = os.path.join(config.data_path(), "right")


"Inception preprocessing which converts the RGB values from [0, 255] to [-1, 1]"
def preprocess_input(x):
    x = np.divide(x, 255)
    x = np.subtract(x, 0.5)
    x = np.multiply(x, 2)
    return np.array(x)



## rescaling with keeping aspect ratio
def get_image(path):
    img_resize = Image.open(path)
    width, height = img_resize.size  # Pillow return images size as (w, h)

    size = config.IMG_SIZE
    if(width > height):
        new_width = size
        new_height = int(size * (height / width) + 0.5)

    else:
        new_height = size
        new_width = int(size * (width / height) + 0.5)


    #resize for keeping aspect ratio
    img_res = img_resize.resize((new_width, new_height), resample = PIL.Image.BICUBIC)

    #Pad the borders to create a square image
    img_pad = Image.new("RGB", (size, size), (0, 0, 0))
    ulc = ((size - new_width) // 2, (size - new_height) // 2)
    img_pad.paste(img_res, ulc)

    return img_pad 


def prep_images(images):
    
    '''images is the list of images with absolute path
    '''
    data = np.ndarray((len(images), config.IMG_SIZE, config.IMG_SIZE, config.IMG_CHANNEL),
                      dtype = np.float32)

    # Pillow returns numpy array of (width, height,channel(RGB))
    for i, image_file in enumerate(images):
        img = get_image(image_file)

        img_px = np.array(img)  #convert PIL image to numpy array
        data[i] = preprocess_input(img_px)

        if (i % 1000 == 0):
            print("loading %s images" % i)

    return data




def process_train_images():
    
    #normal eye images
    l_images = [os.path.join(left_img_dir, img) for img  in os.listdir(left_img_dir)]
    
    #labels of normal eye images
    l_images_label = [[0, 1] for i in range(0,  int(len(os.listdir(left_img_dir))))]


    #diseased eye images
    r_images = [os.path.join(right_img_dir, img) for img  in os.listdir(right_img_dir) ]
    
    #labels of diseased  eye images
    r_images_label = [[1, 0] for i in range(0,  int(len(os.listdir(right_img_dir))))]
    

    #combine all the labels
    l_images_label.extend(r_images_label)
    
    #combine all the images
    l_images.extend(r_images)

    return prep_images(l_images), np.array(l_images_label)
    


if __name__ == "__main__":
    process_train_images()
