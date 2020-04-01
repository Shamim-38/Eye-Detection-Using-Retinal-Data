"""Our TeaNet for diabetic retinopathy eye classification task"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# python moduels
import os
from keras import layers, Input, Model, backend
from keras.layers import Dense, ZeroPadding2D, Conv2D, MaxPooling2D, Flatten, LeakyReLU, Dropout
from keras import regularizers


# project modules
from .. import config



# main method here
def TeaNet(classes = 2):
    img_input = layers.Input(shape = (config.IMG_SIZE, config.IMG_SIZE, 3)) #(256, 256 3)
    """
    # first block
    x = Conv2D(32, (4, 4), strides = 2, padding = "same",
            kernel_regularizer = regularizers.l1(0.00002)) (img_input)  #(256, 256, 32) downsample 
    x = ZeroPadding2D(padding = (1, 1)) (x)
    """
    x = Conv2D(32, (3, 3), padding='same', activation='relu') (img_input)
    
    x = MaxPooling2D(pool_size = (2, 2)) (x)
    x = Dropout(0.2) (x)

    # second block
    x = Conv2D(64, (3, 3), padding='same', activation='relu') (x)
    x = MaxPooling2D(pool_size = (2, 2)) (x)   #(127, 127, 32) downsample
    x = Dropout(0.25) (x)
    
    x = Conv2D(128, (3, 3), padding='same', activation='relu') (x)
    x = MaxPooling2D(pool_size = (2, 2)) (x)   #(127, 127, 32) downsample
    x = Dropout(0.25) (x)
    

    """
    # top layer
    x = Flatten()(x) 
    x = Dense(128, activation='relu', kernel_initializer='he_uniform')(x)
    x = Dropout(0.2) (x)
    """

    x = Flatten()(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(2, activation='softmax')(x)    


    # create model.
    model = Model(inputs = img_input, outputs = x)
    return model


if __name__ == "__main__":
    model = TeaNet()
    model.summary()