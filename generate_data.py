# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 16:02:55 2023

@author: Eoin.Walsh
"""

import lacey_carbon_TEM_synthetic_data_generator as generator 


''' Generate synthetic lacey carbon training data for the synthetic data paper.
    
    Generated data saved in a new folder called 'synthetic_data'.

    Can use "small mag", "medium mag", or "large mag" as string for variable 'dataset_type'.
'''

number_training_images = 30

number_validation_images = 10

dataset_type = "large mag" # can call "small mag", "medium mag", or "large mag"

generator.choose_dataset(dataset_type, number_training_images, number_validation_images)
