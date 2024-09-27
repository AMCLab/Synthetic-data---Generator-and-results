# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 10:41:18 2023

@author: eoin.walsh
"""

import numpy as np
import tifffile as tif
import os
import csv
from sklearn.metrics import precision_score, f1_score, recall_score
from tqdm import tqdm

def create_dir(dir):
    
    ## if the path doesn't exist, make it
    if os.path.isdir(dir) != True:
        
        os.mkdir(dir) # make the path

def normalise(array):
    #change the range of an array for all values so they lie between 0 & 1. 
    array_norm = ((array - np.min(array)) / ((np.max(array)) - np.min(array))).astype("float32")
    
    return array_norm

os.chdir("..")

create_dir("experimental data/amorphous carbon grids/threshold_algorithm_analysis")

save_path = "experimental data/amorphous carbon grids/threshold_algorithm_analysis"

manual_path = "experimental data/amorphous carbon grids/manual_annotations"

threshold_path = "experimental data/amorphous carbon grids/best_thresholds"

for image in tqdm(os.listdir(threshold_path)):
    
    filename = image.split("__")[0]
    
    try:
        
        manual_img = tif.imread(os.path.join(manual_path, filename+".tiff"))
        
        manual_img = (np.ndarray.flatten(manual_img)/255).astype('uint8')
        
        thresh_path = os.path.join(threshold_path, image)
        
        threshold_img = np.where(tif.imread(os.path.join(threshold_path, image)) == 0, 1, 0)
        
        threshold_img = np.ndarray.flatten(threshold_img)
        
        precision_threshold = precision_score(manual_img, threshold_img)
        
        recall_threshold = recall_score(manual_img, threshold_img)
        
        f1_score_threshold = f1_score(manual_img,threshold_img)

        with open(os.path.join(save_path, filename+".csv"), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(zip(["Precision"], ["Recall"], ['F1 Score']))
            
        with open(os.path.join(save_path, filename+".csv"), 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(zip([precision_threshold], [recall_threshold], [f1_score_threshold]))
    except:
        continue