# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 12:00:13 2023

@author: eoin.walsh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile as tif
import os
import csv
from tqdm import tqdm
import ncempy.io.dm as dm
import pandas as pd

def get_resolution(resolution,unit):
    
    if unit == 'nm':
        
        resolution = resolution/1000
        
    return resolution

def normalise(array):
    #change the range of an array for all values so they lie between 0 & 1. 
    array_norm = ((array - np.min(array)) / ((np.max(array)) - np.min(array))).astype("float32")
    
    return array_norm

save = True

dm4_path = "experimental data/amorphous carbon grids/2023_02_MicroscopyData"

data_paths = []

for folders in os.listdir(dm4_path):
    
    if 'mag' in folders:
        
        for data in os.listdir(os.path.join(dm4_path, folders)):
            
            data_paths.append(os.path.join(dm4_path, folders, data))
            
path_ml = "experimental data/amorphous carbon grids/deep_learning_analysis_data"

path_threshold = "experimental data/amorphous carbon grids/threshold_algorithm_analysis"

precision_algo_1 = []

recall_algo_1 = []

f1_score_algo_1 = []

precision_algo_2 = []

recall_algo_2 = []

f1_score_algo_2 = []

precision_algo_3 = []

recall_algo_3 = []

f1_score_algo_3 = []

rc_curve_area_1 = []

pr_curve_area_1 = []

rc_curve_area_2 = []

pr_curve_area_2 = []

rc_curve_area_3 = []

pr_curve_area_3 = []

precision_threshold_2 = []

recall_threshold_2 = []

f1_score_threshold_2 = []

precision_threshold_1 = []

recall_threshold_1 = []

f1_score_threshold_1 = []

precision_threshold_3 = []

recall_threshold_3 = []

f1_score_threshold_3 = []
    
for data in tqdm(os.listdir(path_ml)):
    
    if ("roc_curve" in data) or  ("pr_curve" in data):
        continue
    
    for val in data_paths:
        
        if data[:-4] in val:
    
            dm3_file = dm.dmReader(val) #read the dm3 file
            break
        
    resolution = get_resolution(dm3_file['pixelSize'][0],dm3_file['pixelUnit'][0])
    
    with open(os.path.join(path_ml, data), 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        vals = list(reader)
    
    if ('0012' in data) or ('0013' in data)  or ('0015' in data): #um
    
        precision_algo_2.append(float(vals[0][0]))

        recall_algo_2.append(float(vals[0][1]))

        f1_score_algo_2.append(float(vals[0][2]))

        rc_curve_area_2.append(float(vals[0][3]))
        
        pr_curve_area_2.append(float(vals[0][4]))
        
    elif ('0005' in data) or ('0006' in data) or ('0008' in data): #um
    
        precision_algo_1.append(float(vals[0][0]))

        recall_algo_1.append(float(vals[0][1]))

        f1_score_algo_1.append(float(vals[0][2]))

        rc_curve_area_1.append(float(vals[0][3]))
        
        pr_curve_area_1.append(float(vals[0][4]))
        
    elif ('0016' in data) or ('0018' in data) or ('0019' in data): #um
    
        precision_algo_3.append(float(vals[0][0]))

        recall_algo_3.append(float(vals[0][1]))

        f1_score_algo_3.append(float(vals[0][2]))

        rc_curve_area_3.append(float(vals[0][3]))
        
        pr_curve_area_3.append(float(vals[0][4]))

for data in tqdm(os.listdir(path_threshold)):
    
    if ("roc_curve" in data) or  ("pr_curve" in data):
        continue
    
    for val in data_paths:
        
        if data[:-4] in val:
    
            dm3_file = dm.dmReader(val) #read the dm3 file
            break
        
    resolution = get_resolution(dm3_file['pixelSize'][0],dm3_file['pixelUnit'][0])
    
    with open(os.path.join(path_threshold, data), 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        vals = list(reader)
    
    if ('0012' in data) or ('0013' in data)  or ('0015' in data): #um
    
        precision_threshold_2.append(float(vals[0][0]))

        recall_threshold_2.append(float(vals[0][1]))

        f1_score_threshold_2.append(float(vals[0][2]))

    elif ('0005' in data) or ('0006' in data) or ('0008' in data): #um
    
        precision_threshold_1.append(float(vals[0][0]))

        recall_threshold_1.append(float(vals[0][1]))

        f1_score_threshold_1.append(float(vals[0][2]))

    elif ('0016' in data) or ('0018' in data) or ('0019' in data): #um
    
        precision_threshold_3.append(float(vals[0][0]))

        recall_threshold_3.append(float(vals[0][1]))

        f1_score_threshold_3.append(float(vals[0][2]))

precisions_thresh = [precision_threshold_1, precision_threshold_2, precision_threshold_3]

recalls_thresh = [recall_threshold_1, recall_threshold_2, recall_threshold_3]

f1_score_thresh = [f1_score_threshold_1, f1_score_threshold_2, f1_score_threshold_3]

precisions_ml = [precision_algo_1, precision_algo_2, precision_algo_3]

recalls_ml = [recall_algo_1, recall_algo_2, recall_algo_3]

f1_score_ml = [f1_score_algo_1, f1_score_algo_2, f1_score_algo_3]

roc_avgs = [rc_curve_area_1, rc_curve_area_2, rc_curve_area_3]

pr_avgs = [pr_curve_area_1, pr_curve_area_2, pr_curve_area_3]

all_precisions_thresh = []

all_precisions_ml = []

for j in range(len(precisions_thresh)):
    
    all_precisions_thresh.append(np.round(np.sum(precisions_thresh[j])/len(precisions_thresh[j]),2))

for j in range(len(precisions_thresh)):
    
    all_precisions_ml.append(np.round(np.sum(precisions_ml[j])/len(precisions_ml[j]),2))
    
cols = 43
rows = 10

col_size = 10

row_size = 9

#Create a subplot with 2 rows and 1 column
fig = plt.figure(figsize=(cols,rows))

grid = plt.GridSpec(rows,cols,hspace=1)

labels = ["Small Mag", "Medium Mag", "Large Mag"]

x = np.array([0,0.5,1]) 

width = 0.18  # the width of the bars

font_size = 16

ax = fig.add_subplot(grid[row_size*0:row_size*1, (col_size*0):(col_size*1)])

rects1 = ax.bar(x-0.09, all_precisions_ml, width, label='Deep Learning', color = 'cornflowerblue')
rects2 = ax.bar(x+0.09, all_precisions_thresh, width, label='Best Thresholding Method', color = 'lightcoral')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Precision Score', fontsize = font_size+8)
ax.set_xlabel('Data Subset', fontsize = font_size+8)
ax.set_title('Average Precision', fontsize = font_size+16)
ax.set_xticks(x, labels, fontsize = font_size+4)
ax.tick_params(axis='y', labelsize=font_size)
ax.legend(fontsize = font_size + 2, loc = 'upper right')
ax.set_ylim([0,1.19])
ax.set_xlim([-0.3,1.3])
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.bar_label(rects1, all_precisions_ml, padding=0, fontsize = font_size)
ax.bar_label(rects2, all_precisions_thresh, padding=0, fontsize = font_size)

all_recalls_thresh = []

all_recalls_ml = []

for j in range(len(precisions_thresh)):
    
    all_recalls_thresh.append(np.round(np.sum(recalls_thresh[j])/len(recalls_thresh[j]),2))

for j in range(len(precisions_thresh)):
    
    all_recalls_ml.append(np.round(np.sum(recalls_ml[j])/len(recalls_ml[j]),2))
    
ax = fig.add_subplot(grid[row_size*0:row_size*1, (col_size*1)+1:(col_size*2)+1])

rects1 = ax.bar(x-0.09, all_recalls_ml, width, label='Deep Learning', color = 'cornflowerblue')
rects2 = ax.bar(x+0.09, all_recalls_thresh, width, label='Best Thresholding Method', color = 'lightcoral')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Recall Score', fontsize = font_size+8)
ax.set_xlabel('Data Subset', fontsize = font_size+8)
ax.set_title('Average Recall', fontsize = font_size+16)
ax.set_xticks(x, labels, fontsize = font_size+4)
ax.tick_params(axis='y', labelsize=font_size)
ax.legend(fontsize = font_size + 2, loc = 'upper right')
ax.set_ylim([0,1.19])
ax.set_xlim([-0.3,1.3])
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.bar_label(rects1, all_recalls_ml, padding=0, fontsize = font_size)
ax.bar_label(rects2, all_recalls_thresh, padding=0, fontsize = font_size)

all_f1scores_thresh = []

all_f1scores_ml = []

for j in range(len(f1_score_thresh)):
    
    all_f1scores_thresh.append(np.round(np.sum(f1_score_thresh[j])/len(f1_score_thresh[j]),2))

for j in range(len(precisions_thresh)):
    
    all_f1scores_ml.append(np.round(np.sum(f1_score_ml[j])/len(f1_score_ml[j]),2))
    

ax = fig.add_subplot(grid[row_size*0:row_size*1, (col_size*2)+2:(col_size*3)+2])

rects1 = ax.bar(x-0.09, all_f1scores_ml, width, label='Deep Learning', color = 'cornflowerblue')
rects2 = ax.bar(x+0.09, all_f1scores_thresh, width, label='Best Thresholding Method', color = 'lightcoral')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('F1 Score Score', fontsize = font_size+8)
ax.set_xlabel('Data Subset', fontsize = font_size+8)
ax.set_title('Average F1 Score', fontsize = font_size+16)
ax.set_xticks(x, labels, fontsize = font_size+4)
ax.tick_params(axis='y', labelsize=font_size)
ax.legend(fontsize = font_size + 2, loc = 'upper right')
ax.set_ylim([0,1.19])
ax.set_xlim([-0.3, 1.3])
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.bar_label(rects1, all_f1scores_ml, padding=0, fontsize = font_size)
ax.bar_label(rects2, all_f1scores_thresh, padding=0, fontsize = font_size)

roc = []

pr = []

for vals in roc_avgs:
    
    roc.append(np.round(np.mean(vals),2))

for vals in pr_avgs:
    
    pr.append(np.round(np.mean(vals),2))
        
ax = fig.add_subplot(grid[row_size*0:row_size*1, (col_size*3)+3:(col_size*4)+3])

rects1 = ax.bar(x-0.09, roc, width, label='Precision-Recall Curve', color = 'forestgreen')
rects2 = ax.bar(x+0.09, pr, width, label='Receiver Operating Characteristic Curve', color = 'darkorange')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Area Under Curve', fontsize = font_size+8)
ax.set_xlabel('Data Subset', fontsize = font_size+8)
ax.set_title('Area Under PR and ROC Curves', fontsize = font_size+16)
ax.set_xticks(x, labels, fontsize = font_size+4)
ax.tick_params(axis='y', labelsize=font_size)
ax.legend(fontsize = font_size + 2, loc = 'upper right')
ax.set_ylim([0,1.19])
ax.set_xlim([-0.3, 1.3])
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.bar_label(rects1, roc, padding=0, fontsize = font_size)
ax.bar_label(rects2, pr, padding=0, fontsize = font_size)

if save == True:
    
    plt.savefig("amorphous_carbon_metrics.svg", dpi=1000)

plt.show()