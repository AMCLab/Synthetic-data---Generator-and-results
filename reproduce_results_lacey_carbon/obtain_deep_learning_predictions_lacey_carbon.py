# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 09:53:36 2021

@author: Eoin.Walsh
"""

from __future__ import division

import os
import ncempy.io.dm as dm
import time
import numpy as np
import cv2
import deep_learning_models as deep_learning_models
import tifffile as tif
from tqdm import tqdm

from keras.models import Model
from keras.layers import Input, BatchNormalization, Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers import concatenate
from keras.optimizers import Adam
from keras import layers

def create_dir(dir):
    
    ## if the path doesn't exist, make it
    if os.path.isdir(dir) != True:
        
        os.mkdir(dir) # make the path

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

def get_unet():
    
    no_of_layers = 3#hyperparameters.Int("no_of_layers", min_value=1, max_value=5, step=1)
    
    lr =  7e-4#hyperparameters.Float("lr", min_value=1e-5, max_value=1e-3, sampling="log")
    
    kernel_size = 7#hyperparameters.Int("kernel size", min_value=3, max_value=7, step=2)
    
    max_pooling = 8#hyperparameters.Choice("Max Pooling", [2, 4])
    
    n_filters = 3#hyperparameters.Int("filters", min_value=1, max_value=4, step=1)
    
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

def normalise(array):
    
    """ Normalise data between 0 and 1 """
    
    array = ((array-np.min(array))/(np.max(array)-np.min(array)))
    
    return array

def predictions_softmax(model,image):
    
    image = np.squeeze(image[:,:]) # remove any dimensions of size 1
    
    # image = (image*(1./255)).astype('float32') # normalise array
    
    image = np.expand_dims(image,axis=0) # add in dimension in position 1 
    
    image = np.expand_dims(image,axis=3) # add in dimension in position 4

    preds_fibrils = model.predict(image, verbose=0) # obtain model prediction for the image
    
    x = np.squeeze(np.argmax(preds_fibrils,axis=3)).astype('uint8') # prediction with 50:50 threshold
        
    return x,image,preds_fibrils

def get_resolution(resolution,unit):
    
    if unit == 'nm':
        
        resolution = resolution/1000
        
    return resolution

###   Load the data that we wish to process   ###

start = time.time() # start the time

small_mag_model = deep_learning_models.small_magnification_model()

medium_mag_model = deep_learning_models.medium_magnification_model()

large_mag_model = deep_learning_models.large_magnification_model()

count_dir = 0

os.chdir("..")

path = os.path.join(os.getcwd(),'experimental data/lacey carbon grids/dm4') # path to the image data

files = os.listdir(path) # list the image files in the path

count_dir += 1 # start the image count

count = 0   

create_dir('experimental data/lacey carbon grids/deep_learning_predictions')

for image in tqdm(files): # loop through the image files

    #image = test_files[1] # focusing on the 2nd image for the time being 
    load_in = os.path.join(path,image) # path to the image

    if '.dm4' in image:
        
        dm3_file = dm.dmReader(load_in) #read the dm3 file
        
        img = np.array(dm3_file['data']) # get the image array from dm3 file
        
        standard_deviation = np.std(img)
        
        resolution = get_resolution(dm3_file['pixelSize'][0],dm3_file['pixelUnit'][0])

        img = cv2.medianBlur(img, 5)
        
        img = normalise(img).astype('float32')
        
        if resolution < 0.0029 and resolution > 0.001: #um
                
            pred,img,raw_pred = predictions_softmax(medium_mag_model,img) # obtain prediction for the image
            
            final_particles_95 = np.squeeze(np.where(raw_pred[:,:,:,1]>0.95,1,0)) # 95% confidence threshold 
            
            final_particles_raw = np.squeeze(raw_pred[:,:,:,1]) # raw probability array
            
            tif.imwrite(os.path.join("experimental data/lacey carbon grids/deep_learning_predictions", image[:-4])+".tiff", 
                        (np.squeeze(final_particles_raw)*255).astype('uint8'))
            
        elif resolution >= 0.0029: #um

            pred,img,raw_pred = predictions_softmax(small_mag_model,img) # obtain prediction for the image
            
            final_particles_95 = np.squeeze(np.where(raw_pred[:,:,:,1]>0.95,1,0)) # 95% confidence threshold 
            
            final_particles_raw = np.squeeze(raw_pred[:,:,:,1]) # raw probability array
            
            tif.imwrite(os.path.join("experimental data/lacey carbon grids/deep_learning_predictions", image[:-4])+".tiff", 
                        (np.squeeze(final_particles_raw)*255).astype('uint8'))
            
        elif resolution < 0.001 and resolution > 0.00025: #um

            pred,img,raw_pred = predictions_softmax(large_mag_model,img) # obtain prediction for the image
            
            final_particles_95 = np.squeeze(np.where(raw_pred[:,:,:,1]>0.95,1,0)) # 95% confidence threshold 
            
            final_particles_raw = np.squeeze(raw_pred[:,:,:,1]) # raw probability array
            
            tif.imwrite(os.path.join("experimental data/lacey carbon grids/deep_learning_predictions", image[:-4])+".tiff", 
                        (np.squeeze(final_particles_raw)*255).astype('uint8'))
                    
print('\n\nFinished all Images. Time: ',np.round(((time.time()-start)/60),2),' minutes') # print the time taken

