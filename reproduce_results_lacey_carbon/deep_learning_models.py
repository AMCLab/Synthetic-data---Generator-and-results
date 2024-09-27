# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 11:14:09 2021

@author: Eoin.Walsh
"""
from __future__ import division

import os

from keras.models import Model
from keras.layers import Input, BatchNormalization, Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers import concatenate
from keras.optimizers import Adam
from keras import layers


#import image_processing as img_proc 

def conv2d_block(input_tensor, n_filters, kernel_size = 3, batchnorm = True):
    
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

def get_small_mag_model(input_img, n_filters, dropout, batchnorm = True):
    
    """ U-Net Machine Learning Algorithm Architecture in Keras"""
    
    no_of_layers = 3
    
    lr =  7e-4

    kernel_size = 5
    
    max_pooling = 4
    
    n_filters = 4
    
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
    
    return model

def get_medium_mag_model(input_img, n_filters, dropout, batchnorm = True):
   
    no_of_layers = 3
    
    lr =  1e-3
    
    kernel_size = 7
    
    max_pooling = 4
    
    n_filters = 4
    
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

def get_large_mag_model(input_img, n_filters, dropout, batchnorm = True):
    
    """ U-Net Machine Learning Algorithm Architecture in Keras"""
    
    no_of_layers = 3
    
    lr =  7e-4
    
    kernel_size = 7
    
    max_pooling = 8
    
    n_filters = 3
    
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
       
    return model

def small_magnification_model():
    
    model_path = os.path.join(os.getcwd(),'synthetic data final/small_mag_model/PAT4NANO_silica_small_mag_final_validation_weights.h5')
    
    im_width,im_height = 2048,2048 # input image height and width
        
    input_img = Input((im_height, im_width, 1), name='img') # input image initial conditions
        
    model = get_small_mag_model(input_img, n_filters=1, dropout=0, batchnorm = True) # call the unet ml architecture function
                                                   
    model.load_weights(model_path) # load the trained model weights into the architecture
    
    return model

def medium_magnification_model():
    
    model_path = os.path.join(os.getcwd(),'synthetic data final/medium_mag_model/PAT4NANO_silica_medium_mag_final_validation_weights.h5')

    im_width,im_height = 2048,2048 # input image height and width
        
    input_img = Input((im_height, im_width, 1), name='img') # input image initial conditions
        
    model = get_medium_mag_model(input_img, n_filters=1, dropout=0, batchnorm = True) # call the unet ml architecture function
                                                   
    model.load_weights(model_path) # load the trained model weights into the architecture
    
    return model

def large_magnification_model():
    
    model_path = os.path.join(os.getcwd(),'synthetic data final/large_mag_model/PAT4NANO_silica_large_mag_final_validation_weights.h5')
    
    im_width,im_height = 2048,2048 # input image height and width
        
    input_img = Input((im_height, im_width, 1), name='img') # input image initial conditions
        
    model = get_large_mag_model(input_img, n_filters=1, dropout=0, batchnorm = True) # call the unet ml architecture function
                                                   
    model.load_weights(model_path) # load the trained model weights into the architecture
    
    return model
