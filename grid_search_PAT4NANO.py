import os
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-bright")

import tensorflow as tf

from keras.models import Model
from keras.layers import Input, BatchNormalization, Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers import concatenate
from keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
import keras_tuner as kt
from keras_tuner import HyperParameters  
 
def conv2d_block(input_tensor, n_filters, kernel_size = 5, batchnorm = True):
    
    """Function to add 2 convolutional layers with the parameters passed to it"""
    
    ## first convolutional layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = layers.ReLU()(x) # Relu activation function
    
    # second convolutional layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = layers.ReLU()(x) # Relu activation function
    
    return x

def get_unet(hyperparameters):
    
    no_of_layers = hyperparameters.Int("no_of_layers", min_value=1, max_value=3, step=1)
    
    lr =  1e-3#hyperparameters.Float("lr", min_value=5e-4, max_value=1e-3, sampling="log")
    
    kernel_size = hyperparameters.Int("kernel size", min_value=3, max_value=7, step=2)
    
    max_pooling = hyperparameters.Choice("Max Pooling", [2, 4])
    
    n_filters = hyperparameters.Int("filters", min_value=1, max_value=4, step=1)
    
    im_height, im_width = 2048, 2048
    
    input_img = Input((im_height, im_width, 1), name='img') # set the size of the input image arrays

    batchnorm = True
    
    dropout = 0
    
    ml_layers = {}
    
    """ U-Net Machine Learning Algorithm Architecture in Keras"""
    
    # Contracting Path
    
    for i in range(no_of_layers):
        
        if i == 0:
            
            ## 1st encoding layer 
            ml_layers['cd'+str(i)] = conv2d_block(input_img, n_filters*1, kernel_size = kernel_size, batchnorm = batchnorm) # convolutional block
            ml_layers['mp'+str(i)] = MaxPooling2D((max_pooling,max_pooling))(ml_layers['cd'+str(i)]) # max pooling layer to decrease size of convolutional block output
            ml_layers['p'+str(i)] = Dropout(dropout)(ml_layers['mp'+str(i)]) # random parameter dropoout layer, to help prevent overfitting, not used if dropout = 0

        elif i > 0:
            
            ml_layers['cd'+str(i)] = conv2d_block(ml_layers['p'+str(i-1)], n_filters*1, kernel_size = kernel_size, batchnorm = batchnorm) # convolutional block
            ml_layers['mp'+str(i)] = MaxPooling2D((max_pooling,max_pooling))(ml_layers['cd'+str(i)]) # max pooling layer to decrease size of convolutional block output
            ml_layers['p'+str(i)] = Dropout(dropout)(ml_layers['mp'+str(i)]) # random parameter dropoout layer, to help prevent overfitting, not used if dropout = 0
    
    ## Intermediate layer between encoder and decoder
    ml_layers['mid'] = conv2d_block(ml_layers['p'+str(no_of_layers-1)], n_filters = n_filters * 2, kernel_size = kernel_size, batchnorm = batchnorm)
    
    # Expanding Path
    
    for j in range(no_of_layers):
        
        if j == (no_of_layers-1) and (j != 0):
            
            ml_layers['u'+str(j)] = Conv2DTranspose(n_filters * 1, kernel_size = kernel_size, strides = (max_pooling,max_pooling), padding = 'same')(ml_layers['cu'+str(j-1)])
            ml_layers['u'+str(j)] = concatenate([ml_layers['u'+str(j)], ml_layers['cd'+str(i-j)]])
            ml_layers['u'+str(j)] = Dropout(dropout)(ml_layers['u'+str(j)])
            ml_layers['cu'+str(j)] = conv2d_block(ml_layers['u'+str(j)], n_filters * 2, kernel_size = kernel_size, batchnorm = batchnorm)
            
        elif j == 0:
            
            ml_layers['u'+str(j)] = Conv2DTranspose(n_filters * 1, kernel_size = kernel_size, strides = (max_pooling,max_pooling), padding = 'same')(ml_layers['mid'])
            ml_layers['u'+str(j)] = concatenate([ml_layers['u'+str(j)], ml_layers['cd'+str(i-j)]])
            ml_layers['u'+str(j)] = Dropout(dropout)(ml_layers['u'+str(j)])
            ml_layers['cu'+str(j)] = conv2d_block(ml_layers['u'+str(j)], n_filters * 1, kernel_size = kernel_size, batchnorm = batchnorm)
            
        elif j >= 0:
            
            ml_layers['u'+str(j)] = Conv2DTranspose(n_filters * 1, kernel_size = kernel_size, strides = (max_pooling,max_pooling), padding = 'same')(ml_layers['cu'+str(j-1)])
            ml_layers['u'+str(j)] = concatenate([ml_layers['u'+str(j)], ml_layers['cd'+str(i-j)]])
            ml_layers['u'+str(j)] = Dropout(dropout)(ml_layers['u'+str(j)])
            ml_layers['cu'+str(j)] = conv2d_block(ml_layers['u'+str(j)], n_filters * 1, kernel_size = kernel_size, batchnorm = batchnorm)
            
    #output layer
    outputs = Conv2D(2, (1, 1), activation='softmax')(ml_layers['cu'+str(j)]) #softmax activation function at the output layer
    
    model = Model(inputs=[input_img], outputs=[outputs])
    
    model.compile(optimizer=Adam(lr=lr), loss="sparse_categorical_crossentropy", metrics=["sparse_categorical_accuracy"]) # compile the algorithm along with setting optimixer plus the loss function
        
    return model

im_width, im_height = 2048, 2048

image_datagen = ImageDataGenerator() # keras object that allows for easy manipulation of images in folders
mask_datagen = ImageDataGenerator() # keras object that allows for easy manipulation of images in folders

batch_size = 2 # number of images passed through ML algorithm during training iteration

seed1 = np.random.randint(0,10000) # random seed for image shuffle
seed2 = np.random.randint(0,10000) # random seed for image shuffle

image_generator = image_datagen.flow_from_directory(directory='synthetic_data/training_data/imgs', # folder files are in
                                             class_mode=None, # how many data classes are there ? (Used for image classification rather than segmentation)
                                             shuffle=True, # shuffle the images
                                             seed=seed1, # random seed for shuffling
                                             batch_size =batch_size, # number of images per training iteration
                                             target_size=(im_width,im_height), # image sizes (can resize here)
                                             color_mode="grayscale") #number of image channels (grayscale/rgb)

mask_generator = mask_datagen.flow_from_directory(directory='synthetic_data/training_data/msks',
                                             class_mode=None,
                                             shuffle=True,
                                             seed=seed1,
                                             batch_size =batch_size,
                                             target_size=(im_width,im_height),
                                             color_mode="grayscale")

training_generator = zip(image_generator, mask_generator) # zip training images and masks into one generator object

image_generator1 = image_datagen.flow_from_directory(directory='synthetic_data/validation_data/imgs',
                                              class_mode=None,
                                              shuffle=True,
                                              seed=seed2,
                                              batch_size =batch_size,
                                              color_mode="grayscale",
                                              target_size=(im_width,im_height))
                                             
mask_generator1 = mask_datagen.flow_from_directory(directory='synthetic_data/validation_data/msks',
                                              class_mode=None,
                                              shuffle=True,
                                              seed=seed2,
                                              batch_size =batch_size,
                                              color_mode="grayscale",
                                              target_size=(im_width,im_height))

validation_generator = zip(image_generator1, mask_generator1) #zip validation images and masks into one generator object

hp = HyperParameters()

get_unet(hp)

epochs = 25

num_trials = 50
      
tuner = kt.RandomSearch(hypermodel=get_unet,
                        objective='val_loss',
                        max_trials=num_trials,
                        executions_per_trial=1,
                        overwrite=True,
                        directory=os.getcwd(),
                        project_name="tune_hyperparameters")

callbacks = [        
            EarlyStopping(monitor="val_loss",patience=25,verbose=1)# stop training if loss doesn't decrease over set time (patience)
]

#define the number of images in the training data
no_of_training_images = int(len(os.listdir('synthetic_data/training_data/imgs/images')))
        
#define the number of images in the validation data
no_of_validation_images = int(len(os.listdir('synthetic_data/validation_data/imgs/images')))
    
tuner.search(training_generator, #training data
             steps_per_epoch = int((no_of_training_images)/(batch_size*5)), #training steps
             epochs=epochs, #epochs in full run
             validation_data = validation_generator, #validation data
             validation_steps = int((no_of_validation_images)/(batch_size*5)), #validation steps
             callbacks=callbacks) #callbacks

tuner.results_summary(num_trials=3)

print('Done')
