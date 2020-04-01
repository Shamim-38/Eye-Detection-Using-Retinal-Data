# python packages
import numpy as np
import os
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator


# project modules
from .. import config
from . import _preparation
from . import model_utils
from . import eye_classification_model as ecm


# loading data

X_train, y_train = _preparation.process_train_images()
print("Trian Data: ", X_train.shape)
print("Train Lable: ", len(y_train))


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


history = model.fit (X_train, y_train,
                            epochs = config.epochs,
                            callbacks=[model_cp, reduce_lr, early_stopping],
                            validation_split = 0.15,
                            batch_size = config.batch_size,
                            verbose = 2)

print(history)


# saving model                  
model_utils.save_model(model, model_utils.MODEL_ED, model_utils.WEIGHT_ED)
