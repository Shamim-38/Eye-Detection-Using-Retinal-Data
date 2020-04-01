""" this file preprocess retinal fundus images """

# python modules
import cv2, glob, numpy
import os
import random

# project modules
from .. import config



#Scale 300 seems to be sufficient; 500 and 1000 may be overkill
scale = 300
def scaleRadius(img, scale):
    
    x = img[int(img.shape[0]/2),:,:].sum(1)
    r = (x>x.mean()/10).sum()/2
    s = scale * 1.0 / r    

    return cv2.resize(img,(0,0),fx=s,fy=s)



def preprocess_ball(input_path, image_list, output_dir):
    for f in image_list:
        try:
            s = os.path.join(input_path, f)
            a = cv2.imread(s)
            a = scaleRadius(a, scale)
            
            b = numpy.zeros(a.shape)
            cv2.circle(b,(int(a.shape[1]/2), int(a.shape[0]/2)),
                       int(scale*0.9),(1,1,1), -1,8,0)
            
            aa = cv2.addWeighted(a, 4, cv2.GaussianBlur(a, (0,0), scale / 30), -4, 128)*b + 128 * (1-b)
            cv2.imwrite(os.path.join(output_dir, f), aa)

        except:
            pass



# main method
def do_preprocess(class_id):

    input_path = os.path.join(config.dataset_path(), class_id)
    image_list = os.listdir(input_path)


    output_path = os.path.join(config.data_path(), class_id)
    os.makedirs(output_path, exist_ok = True)

    print("images are preprocessing ...")
    preprocess_ball(input_path, image_list, output_path)


# calling 
do_preprocess("right")