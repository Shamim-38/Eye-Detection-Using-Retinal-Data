# python packages
import numpy as np
import os
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator


# project modules
from .. import config
from . import model_utils
from . import eye_classification_model as ecm



def total_count_files(folder_dir):
    count = 0
    for dir in os.listdir(folder_dir):
        count += len(os.listdir((os.path.join(folder_dir, dir))))

    return count



# path variables and constans
TRAIN_DIR = os.path.join(config.data_path(), "train")
VALID_DIR = os.path.join(config.data_path(), "validation")


# calculation for train data
total_train_images = total_count_files(TRAIN_DIR)
steps_train = int(total_train_images // config.batch_size) + 1

#calculation for validation data
total_val_images = total_count_files(VALID_DIR)
steps_val = int(total_val_images // config.batch_size) + 1


print("total training images", total_train_images)
print("total training images", total_val_images)


# generating data
train_data_gen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)
    
val_data_gen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)


#train data generator
train_data = train_data_gen.flow_from_directory(
    TRAIN_DIR,
    classes = ["left", "right"],
    target_size = (config.IMG_SIZE, config.IMG_SIZE),
    batch_size = config.batch_size,
    class_mode = 'categorical')


#validation data generator
val_data = val_data_gen.flow_from_directory(
    VALID_DIR,
    classes = ["left", "right"],
    target_size = (config.IMG_SIZE, config.IMG_SIZE),
    batch_size = config.batch_size,
    shuffle = False,
    class_mode = 'categorical')


################# Eye Classification Network #################
print("loading model")
model = ecm.TeaNet(classes = config.ed_num_classes)


optimizer_sgd = SGD(lr = config.l_rate, momentum = 0.9)
objective = "categorical_crossentropy"


print("compiling model")
model.compile(optimizer = optimizer_sgd,
              loss = objective,
              metrics = ['accuracy'])


#To train using pretrained weight
#model.load_weights(os.path.join(config.output_path(), model_utils.WEIGHT_ED))


# callbacks
early_stopping = model_utils.set_early_stopping()
model_cp = model_utils.set_model_checkpoint(model_utils.WEIGHT_ED)
reduce_lr = model_utils.set_reduce_lr()


hiistory = model.fit_generator(train_data,
                            steps_per_epoch = steps_train,
                            epochs = config.epochs,
                            callbacks=[model_cp, reduce_lr, early_stopping],
                            validation_data = val_data,
                            validation_steps = steps_val,
                            verbose = 2)

print(history)


# saving model                  
model_utils.save_model(model, model_utils.MODEL_ED, model_utils.WEIGHT_ED)