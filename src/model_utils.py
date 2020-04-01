
"""this file contains code for handling  model utilities"""
""" Notes:
# all final models are palced and stored manually in weight directory
# model checkpoint are saved in output directory
"""

# python modules
from keras.callbacks import (EarlyStopping,
                             Callback,
                             ModelCheckpoint,
                             ReduceLROnPlateau)

from keras.models import model_from_json
import numpy
import os

# project modules
from .. import config



####################### ED Network #######################
#model name
MODEL_ED = "ed_model.json"
WEIGHT_ED = "ed_weight.h5"




# save model to output directory
def save_model(model, model_name, weight_name):

    #serialize the model to json
    model_json = model.to_json()

    #write
    with open(os.path.join(config.output_path(), model_name), "w") as json_file:
        json_file.write(model_json)
        
    print("model saved")

    #save the weight to HDF5
    model.save_weights(os.path.join(config.output_path(), weight_name))
    print("weight saved")
    


# load a model from output directory by name
def load_model_only(model_name):

    json_file = open(os.path.join(config.output_path(), model_name))
    loaded_model = model_from_json(json_file.read()) 
   
    json_file.close()
    return loaded_model



# load a model with weight from output directory by name
def load_model(model_name, weight_name):

    json_file = open(os.path.join(config.output_path(), model_name))
    loaded_model = model_from_json(json_file.read())
    
    json_file.close()
    print("model loaded")
    
    loaded_model.load_weights(os.path.join(config.output_path(), weight_name))
    print("weight loaded")
    
    return loaded_model




def save_model_only(model, model_name):
    model_json = model.to_json()
    with open(os.path.join(config.output_path(), model_name),"w") as json_file:
        json_file.write(model_json)
        print("model saved")





# for early stopping if no improvement within patience batches occured
def set_early_stopping():
    return EarlyStopping(monitor="val_loss",
                         patience = 25,
                         mode = "auto",
                         verbose = 2)



def set_model_checkpoint(weight_name):
    return ModelCheckpoint(os.path.join(config.output_path(), weight_name),
                monitor = 'val_loss',
                verbose = 2,
                save_best_only = True,
                save_weights_only = True,
                mode = 'auto',
                period = 2)



def set_reduce_lr():
    return ReduceLROnPlateau(monitor='val_loss',
                             factor = 0.5,
                             patience = 5,
                             min_lr = 1e-6)