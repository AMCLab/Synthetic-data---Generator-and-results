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

os.chdir("..")

manual_path = "experimental data/lacey carbon grids/manual_annotations"

threshold_path = "experimental data/lacey carbon grids/best_global_threshold_algorithm_per_image"

create_dir('experimental data/lacey carbon grids/threshold_algorithm_analysis')

save_path = "experimental data/lacey carbon grids/threshold_algorithm_analysis"

for image in tqdm(os.listdir(threshold_path)):
    
    filename = image.split("__")[0]
    
    manual_img = tif.imread(os.path.join(manual_path, filename+".tiff"))
    
    manual_img = (np.ndarray.flatten(manual_img)/255).astype('uint8')
    
    thresh_path = os.path.join(threshold_path, image)
    
    threshold_img = np.where(tif.imread(thresh_path)==255,0,255).astype('uint8')

    threshold_img = np.ndarray.flatten(threshold_img)/255

    precision_threshold = precision_score(manual_img, threshold_img)
    
    recall_threshold = recall_score(manual_img, threshold_img)
    
    f1_score_threshold = f1_score(manual_img,threshold_img)

    with open(os.path.join(save_path, filename+".csv"), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(zip(["Precision"], ["Recall"], ['F1 Score']))
        
    with open(os.path.join(save_path, filename+".csv"), 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(zip([precision_threshold], [recall_threshold], [f1_score_threshold]))
   