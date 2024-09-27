# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 10:41:18 2023

@author: eoin.walsh
"""

import numpy as np
import tifffile as tif
import os
import csv
from sklearn.metrics import precision_score, f1_score, precision_recall_curve, recall_score, roc_curve, roc_auc_score, average_precision_score
from tqdm import tqdm
import cv2

def create_dir(dir):
    
    ## if the path doesn't exist, make it
    if os.path.isdir(dir) != True:
        
        os.mkdir(dir) # make the path

os.chdir("..")

saving_path = 'experimental data/lacey carbon grids/deep_learning_analysis'

create_dir(saving_path)

manual_path = "experimental data/lacey carbon grids/manual_annotations"

ml_raw_path = "experimental data/lacey carbon grids/deep_learning_predictions"

for image in tqdm(os.listdir(manual_path)):
    
    manual_img = tif.imread(os.path.join(manual_path, image))
    
    manual_img = np.ndarray.flatten(manual_img)/255
    
    ml_raw_img = tif.imread(os.path.join(ml_raw_path, image))
    
    ml_otsu_img = cv2.threshold(ml_raw_img ,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    
    ml_otsu_img = np.ndarray.flatten(ml_otsu_img)/255
    
    ml_raw_img = np.ndarray.flatten(ml_raw_img)/255
    
    precision_otsu = precision_score(manual_img, ml_otsu_img)
    
    recall_otsu = recall_score(manual_img, ml_otsu_img)
    
    f1_score_otsu = f1_score(manual_img, ml_otsu_img)
    
    pr_curve = precision_recall_curve(manual_img, np.round(ml_raw_img,2))
    
    roc_curves = roc_curve(manual_img, np.round(ml_raw_img,2), drop_intermediate = False)
    
    roc_auc_area = roc_auc_score(manual_img, ml_raw_img)
    
    pr_curve_area = average_precision_score(manual_img, ml_raw_img)
    
    with open(os.path.join(saving_path, image[:-5]+".csv"), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(zip(["Precision"], ["Recall"], ['F1 Score'], ["Area Under ROC Curve"], ["Area Under PR Curve"]))
        
    with open(os.path.join(saving_path, image[:-5]+".csv"), 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(zip([precision_otsu], [recall_otsu], [f1_score_otsu], [roc_auc_area], [pr_curve_area]))
   
    with open(os.path.join(saving_path, image[:-5]+"_roc_curve.csv"), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(zip(["False Positive Rate"], ["True Positive Rate"], ['Thresholds']))
                              
    with open(os.path.join(saving_path, image[:-5]+"_roc_curve.csv"), 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(zip(roc_curves[0].tolist(), roc_curves[1].tolist(), roc_curves[2].tolist()))
    
    with open(os.path.join(saving_path, image[:-5]+"_pr_curve.csv"), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(zip(["Precision"], ["Recall"], ['Thresholds']))
           
    with open(os.path.join(saving_path, image[:-5]+"_pr_curve.csv"), 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(zip(pr_curve[0].tolist(), pr_curve[1].tolist(), pr_curve[2].tolist()))