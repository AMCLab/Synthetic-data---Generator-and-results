# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 16:02:55 2023

@author: Eoin.Walsh
"""

import lacey_carbon_TEM_synthetic_data_generator as generator 
import numpy as np

''' Script for experimenting with parameters in the lacey carbon grid generator codebase.
'''

generator.generate_data(number_images=1, #number of images to generate in total.
                        number_particles=np.random.randint(5,8), #range of number of particles to generate per image, randomised in this range per image.
                        magnification_range=(10000,20000), #magnification range for the generated dataset.
                        number_magnification_intervals=3, #number of magnification intervals in the defined range, not used if 'rand_mag=False' below.
                        particle_intensity_range=(-3, -4.1), #particle pixel intensity range
                        background_brightness_range=(-0.2, 0.2), #image background brightness range for a dataset, randomly selected per image from the range.
                        graphene_intensity_range=(-0.9, -0.7, 0.1), #intensity of the graphene brightness for a dataset, randomly selected per image from the range.
                        lacey_carbon_intensity_mean_range=(-3, -2.5), #mean value range for lacey carbon brightness
                        lacey_carbon_intensity_std=0.1, #standard deviation range for the lacey carbon brightness
                        random_lacey_carbon_brightness_change=False, #random change of lacey carbon brightness for a small section within an image.
                        gradient=True, #gradient change in brightness for a particle on/off
                        shadow=True,  #shadow around a particle on/off
                        graphene_intensity_mean_rand = True, #mean of the graphene brightness selected at random on/off
                        include_sobel_graphene = True, #include sobel filter for the graphene, giving a 3-d effect on/off
                        rand_mag = True, #random magnification selected from range defined using 'magnification_range'
                        folder='training_data', #training data or validation data.
                        plot_image=True, #plot generated image & mask true/false.
                        save_image=False) #save generated image & mask true/false
