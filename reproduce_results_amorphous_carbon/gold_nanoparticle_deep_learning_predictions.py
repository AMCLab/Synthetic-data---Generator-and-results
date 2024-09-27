# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 16:55:17 2022

@author: Eoin.Walsh
"""

import os
import numpy as np
import ncempy.io.dm as dm
import cv2
from tqdm import tqdm
from keras.models import Model
from keras.layers import Input, BatchNormalization, Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers import concatenate
from keras.optimizers import Adam
from keras import layers
import tifffile as tif

# import particle_measurer_data_driven as pm

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

def get_small_unet(input_img, n_filters=1, dropout=0, batchnorm = True):
    
    """ U-Net Machine Learning Algorithm Architecture in Keras"""
   
    no_of_layers = 2#hyperparameters.Int("no_of_layers", min_value=1, max_value=5, step=1)
    
    lr =  4.37e-5 #hyperparameters.Float("lr", min_value=5e-4, max_value=1e-3, sampling="log")
    
    kernel_size = 5#hyperparameters.Int("kernel size", min_value=3, max_value=7, step=2)
    
    max_pooling = 2#hyperparameters.Choice("Max Pooling", [2, 4])
    
    n_filters = 2#hyperparameters.Int("filters", min_value=1, max_value=4, step=1)
    
    im_height, im_width = 2048, 2048
    
    input_img = Input((im_height, im_width, 1), name='img') # set the size of the input image arrays
    
    batchnorm = True
    
    dropout = 0 #hyperparameters.Float("dropout", min_value=0.0, max_value=0.3)
    
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
            
            ml_layers['cd'+str(i)] = conv2d_block(ml_layers['p'+str(i-1)], n_filters**(i+1), kernel_size = kernel_size, batchnorm = batchnorm) # convolutional block
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
            
            ml_layers['u'+str(j)] = Conv2DTranspose(n_filters**(i+1), kernel_size = kernel_size, strides = (max_pooling,max_pooling), padding = 'same')(ml_layers['cu'+str(j-1)])
            ml_layers['u'+str(j)] = concatenate([ml_layers['u'+str(j)], ml_layers['cd'+str(i-j)]])
            ml_layers['u'+str(j)] = Dropout(dropout)(ml_layers['u'+str(j)])
            ml_layers['cu'+str(j)] = conv2d_block(ml_layers['u'+str(j)], n_filters**(i+1), kernel_size = kernel_size, batchnorm = batchnorm)
            
    #output layer
    outputs = Conv2D(2, (1, 1), activation='softmax')(ml_layers['cu'+str(j)]) #softmax activation function at the output layer
    
    model = Model(inputs=[input_img], outputs=[outputs])
    
    model.compile(optimizer=Adam(lr=lr), loss="binary_crossentropy", metrics=["binary_accuracy"]) # compile the algorithm along with setting optimixer plus the loss function
        
    return model

def get_medium_unet(input_img, n_filters=1, dropout=0, batchnorm = True):
    
    """ U-Net Machine Learning Algorithm Architecture in Keras"""
    
    no_of_layers = 3#hyperparameters.Int("no_of_layers", min_value=1, max_value=5, step=1)
    
    lr =  4.37e-5 #hyperparameters.Float("lr", min_value=5e-4, max_value=1e-3, sampling="log")
    
    kernel_size = 5#hyperparameters.Int("kernel size", min_value=3, max_value=7, step=2)
    
    max_pooling = 2#hyperparameters.Choice("Max Pooling", [2, 4])
    
    n_filters = 3#hyperparameters.Int("filters", min_value=1, max_value=4, step=1)
    
    im_height, im_width = 2048, 2048
    
    input_img = Input((im_height, im_width, 1), name='img') # set the size of the input image arrays
    
    batchnorm = True
    
    dropout = 0 #hyperparameters.Float("dropout", min_value=0.0, max_value=0.3)
    
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
            
            ml_layers['cd'+str(i)] = conv2d_block(ml_layers['p'+str(i-1)], n_filters**(i+1), kernel_size = kernel_size, batchnorm = batchnorm) # convolutional block
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
            
            ml_layers['u'+str(j)] = Conv2DTranspose(n_filters**(i+1), kernel_size = kernel_size, strides = (max_pooling,max_pooling), padding = 'same')(ml_layers['cu'+str(j-1)])
            ml_layers['u'+str(j)] = concatenate([ml_layers['u'+str(j)], ml_layers['cd'+str(i-j)]])
            ml_layers['u'+str(j)] = Dropout(dropout)(ml_layers['u'+str(j)])
            ml_layers['cu'+str(j)] = conv2d_block(ml_layers['u'+str(j)], n_filters**(i+1), kernel_size = kernel_size, batchnorm = batchnorm)
            
    #output layer
    outputs = Conv2D(2, (1, 1), activation='softmax')(ml_layers['cu'+str(j)]) #softmax activation function at the output layer
    
    model = Model(inputs=[input_img], outputs=[outputs])
    
    model.compile(optimizer=Adam(lr=lr), loss="binary_crossentropy", metrics=["binary_accuracy"]) # compile the algorithm along with setting optimixer plus the loss function
        
    return model

def get_large_unet(input_img, n_filters=1, dropout=0, batchnorm = True):
    
    """ U-Net Machine Learning Algorithm Architecture in Keras"""
    
    no_of_layers = 3#hyperparameters.Int("no_of_layers", min_value=1, max_value=5, step=1)
    
    lr =  4.37e-5 #hyperparameters.Float("lr", min_value=5e-4, max_value=1e-3, sampling="log")
    
    kernel_size = 5#hyperparameters.Int("kernel size", min_value=3, max_value=7, step=2)
    
    max_pooling = 2#hyperparameters.Choice("Max Pooling", [2, 4])
    
    n_filters = 3#hyperparameters.Int("filters", min_value=1, max_value=4, step=1)
    
    im_height, im_width = 2048, 2048
    
    input_img = Input((im_height, im_width, 1), name='img') # set the size of the input image arrays
    
    batchnorm = True
    
    dropout = 0 #hyperparameters.Float("dropout", min_value=0.0, max_value=0.3)
    
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
            
            ml_layers['cd'+str(i)] = conv2d_block(ml_layers['p'+str(i-1)], n_filters**(i+1), kernel_size = kernel_size, batchnorm = batchnorm) # convolutional block
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
            
            ml_layers['u'+str(j)] = Conv2DTranspose(n_filters**(i+1), kernel_size = kernel_size, strides = (max_pooling,max_pooling), padding = 'same')(ml_layers['cu'+str(j-1)])
            ml_layers['u'+str(j)] = concatenate([ml_layers['u'+str(j)], ml_layers['cd'+str(i-j)]])
            ml_layers['u'+str(j)] = Dropout(dropout)(ml_layers['u'+str(j)])
            ml_layers['cu'+str(j)] = conv2d_block(ml_layers['u'+str(j)], n_filters**(i+1), kernel_size = kernel_size, batchnorm = batchnorm)
            
    #output layer
    outputs = Conv2D(2, (1, 1), activation='softmax')(ml_layers['cu'+str(j)]) #softmax activation function at the output layer
    
    model = Model(inputs=[input_img], outputs=[outputs])
    
    model.compile(optimizer=Adam(lr=lr), loss="binary_crossentropy", metrics=["binary_accuracy"]) # compile the algorithm along with setting optimixer plus the loss function
        
    return model

def normalise(array):
    
    """ Normalise data between 0 and 1 """
    
    array = ((array-np.min(array))/(np.max(array)-np.min(array)))
    
    return array

def predictions_softmax(model,image):
    
    image = np.squeeze(image[:,:]) # remove any dimensions of size 1
    
    image = image*(1./255) # normalise array
    
    image = np.expand_dims(image,axis=0) # add in dimension in position 1 
    
    image = np.expand_dims(image,axis=3) # add in dimension in position 4

    preds_fibrils = model.predict(image, verbose=0) # obtain model prediction for the image
    
    x = np.squeeze(np.argmax(preds_fibrils,axis=3)).astype('uint8') # prediction with 50:50 threshold

    return x,image,preds_fibrils

def lower_upper_bound_cut(array):
    
    """ find values that cut the ends off the histogram 
    of the image,in order to make it more viewable and 
    more spread in the 0 to 1 range. """
    
    flatten = np.ndarray.flatten(array) # flatten the array to 1-d
    
    cut_index_lower = int(len(flatten)*0.00001) #find index of value where 0.001% of data is under
    
    cut_index_upper = int(len(flatten)*0.99999) #find index of value where 0.001% of data is above
    
    flatten_sort = np.sort(flatten) # sort the flattened array from lowest to highest
    
    lower_bound_value = flatten_sort[cut_index_lower] #find lower bound value for cut off
    
    upper_bound_value = flatten_sort[cut_index_upper] # find upper bound value for cut off
    
    return lower_bound_value,upper_bound_value

def ml_small_model(model_path):

    im_width,im_height = 2048,2048 # input image height and width
        
    input_img = Input((im_height, im_width, 1), name='img') # input image initial conditions
        
    model = get_small_unet(input_img, n_filters=1, dropout=0, batchnorm = True) # call the unet ml architecture function
                                                   
    model.load_weights(model_path) # load the trained model weights into the architecture
    
    return model

def ml_medium_model(model_path):

    im_width,im_height = 2048,2048 # input image height and width
        
    input_img = Input((im_height, im_width, 1), name='img') # input image initial conditions
        
    model = get_medium_unet(input_img, n_filters=1, dropout=0, batchnorm = True) # call the unet ml architecture function
                                                   
    model.load_weights(model_path) # load the trained model weights into the architecture
    
    return model

def ml_large_model(model_path):

    im_width,im_height = 2048,2048 # input image height and width
        
    input_img = Input((im_height, im_width, 1), name='img') # input image initial conditions
        
    model = get_large_unet(input_img, n_filters=1, dropout=0, batchnorm = True) # call the unet ml architecture function
                                                   
    model.load_weights(model_path) # load the trained model weights into the architecture
    
    return model

def create_dir(dir):
    
    if os.path.isdir(dir) != True:
        
        os.mkdir(dir) # make the path

os.chdir("..")

create_dir("experimental data/amorphous carbon grids/ml otsu threshold")

create_dir("experimental data/amorphous carbon grids/ml raw prediction")

path = os.path.join("experimental data/amorphous carbon grids/2023_02_MicroscopyData")

threshold_save = "experimental data/amorphous carbon grids/ml otsu threshold"

raw_save = "experimental data/amorphous carbon grids/ml raw prediction"

small_model_path = 'reproduce_results_amorphous_carbon/synthetic data final/small mag data/gold_nanoparticles_validation_25th_jan_small_mag.h5'

small_model = ml_small_model(small_model_path)

medium_model_path = 'reproduce_results_amorphous_carbon/synthetic data final/medium mag data/gold_nanoparticles_validation_3rd_april_medium_gold_nano.h5'

medium_model = ml_medium_model(medium_model_path)
 
large_model_path = 'reproduce_results_amorphous_carbon/synthetic data final/large mag data/gold_nanoparticles_validation_23rd_april_large_mag.h5'

large_model = ml_large_model(large_model_path)

for mag in os.listdir(path):

    img_path = os.path.join(path, mag)
    
    files = os.listdir(img_path) # list the image files in the path
    
    same_res = files[:7] + files[8:]
          
    count = 0 # start the image count
    
    for image in tqdm(files):
    
        load_in = os.path.join(img_path,image) # path to the image
        
        if '.dm3' in image:
            
            dm3_file = dm.dmReader(load_in) #read the dm3 file
            
            img_shape = np.array(dm3_file['data']).shape
            
            img = cv2.resize(np.array(dm3_file['data']), dsize=(2048,2048), interpolation= cv2.INTER_NEAREST) # get the image array from dm3 file

            if mag == 'mag_1':

                img = cv2.medianBlur(img, 5)

                img = ((normalise(img))*255).astype('uint8')

                pred,img,raw_pred = predictions_softmax(small_model,img) # obtain prediction for the image
            
            elif mag == 'mag_2':
                
                img = cv2.medianBlur(img, 5)

                img = ((normalise(img))*255).astype('uint8')
            
                pred,img,raw_pred = predictions_softmax(medium_model,img) # obtain prediction for the image

            elif mag == 'mag_3':
            
                img = ((normalise(img))*255).astype('uint8')
            
                pred,img,raw_pred = predictions_softmax(large_model,img) # obtain prediction for the image

            ret3,final_particles_otsu = cv2.threshold(((np.squeeze(raw_pred[0,:,:,1]))*255).astype('uint8'),
                                                      0,
                                                      255,
                                                      cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            
            final_particles_otsu = cv2.resize(final_particles_otsu, 
                                              dsize=(img_shape[1], img_shape[0]), 
                                              interpolation= cv2.INTER_NEAREST)
            
            raw_pred = cv2.resize(np.squeeze(raw_pred[:,:,:,1]), 
                                  dsize=(img_shape[1], img_shape[0]), 
                                  interpolation= cv2.INTER_NEAREST)

            tif.imwrite(os.path.join(threshold_save, image[:-4]+".tiff"), 
                                      (final_particles_otsu).astype('uint8'))
            
            tif.imwrite(os.path.join(raw_save, image[:-4]+".tiff"),
                                     raw_pred)