""" this file loads configurations and paths of the project"""

import os


#####################  image parameters ###############
IMG_SIZE = 180
IMG_CHANNEL = 3
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, IMG_CHANNEL)



################### network parametes ################
ed_num_classes = 2
batch_size = 16
epochs = 40
l_rate = 0.0001


################### path ###########################
def root_path():
    return os.path.dirname(__file__)


def data_path():
    return os.path.join(root_path(),"data")


def dataset_path():
    return os.path.join(root_path(),"dataset")



def src_path():
    return os.path.join(root_path(),"src")


def output_path():
    return os.path.join(root_path(), "output")


def weight_path():
    return os.path.join(root_path(), "weight")