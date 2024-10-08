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

plt.rcParams.update({
    # "text.usetex": True,
    "font.family": "CMU Serif"
})

dm4_path = "data/2023_02_MicroscopyData"

data_paths = []

for folders in os.listdir(dm4_path):
    
    if 'mag' in folders:
        
        for data in os.listdir(os.path.join(dm4_path, folders)):
            
            data_paths.append(os.path.join(dm4_path, folders, data))
            
path_ml = "data/deep_learning_analysis_data"

path_threshold = "data/threshold_algorithm_analysis"

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
    
    if ('0012' in data) or ('0013' in data) or ('0015' in data): #um
    
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
        
    elif ('0017' in data) or ('0018' in data) or ('0019' in data): #um
    
        precision_algo_3.append(float(vals[0][0]))

        recall_algo_3.append(float(vals[0][1]))

        f1_score_algo_3.append(float(vals[0][2]))

        rc_curve_area_3.append(float(vals[0][3]))
        
        pr_curve_area_3.append(float(vals[0][4]))
        
precision_threshold_1 = []

recall_threshold_1 = []

f1_score_threshold_1 = []

precision_threshold_2 = []

recall_threshold_2 = []

f1_score_threshold_2 = []

precision_threshold_3 = []

recall_threshold_3 = []

f1_score_threshold_3 = []

matthews_threshold_3 = []

for data in tqdm(os.listdir(path_threshold)):
    
    for val in data_paths:
        
        if data[:-4] in val:
    
            dm3_file = dm.dmReader(val) #read the dm3 file
            break
        
    resolution = get_resolution(dm3_file['pixelSize'][0],dm3_file['pixelUnit'][0])
    
    with open(os.path.join(path_threshold, data), 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        vals = list(reader)
    
    if ('0012' in data) or ('0013' in data) or ('0015' in data): #um
    
        precision_threshold_2.append(float(vals[0][0]))

        recall_threshold_2.append(float(vals[0][1]))

        f1_score_threshold_2.append(float(vals[0][2]))

    elif ('0005' in data) or ('0006' in data) or ('0008' in data): #um
        
        precision_threshold_1.append(float(vals[0][0]))

        recall_threshold_1.append(float(vals[0][1]))

        f1_score_threshold_1.append(float(vals[0][2]))

    elif ('0017' in data) or ('0018' in data) or ('0019' in data): #um
    
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

for j in range(len(precisions_thresh)):
    
    labels = np.arange(1,len(precisions_ml[j])+1, 1).tolist()
    
    labels = [str(i) for i in labels]
    
    x = np.arange(len(labels))  # the label locations
    
    width = 0.25  # the width of the bars
    
    cols = 21
    rows = 20
    
    col_size = 10
    
    row_size = 6
    
    font_size = 16
    
    #Create a subplot with 2 rows and 1 column
    fig = plt.figure(figsize=(cols,rows))
    
    grid = plt.GridSpec(rows,cols,hspace=1)
    
    ax = fig.add_subplot(grid[row_size*0:row_size*1, col_size*0:col_size*1])
    
    rects1 = ax.bar(x - width/2, precisions_ml[j], width, label='Deep Learning', color = 'cornflowerblue')
    rects2 = ax.bar(x + width/2, precisions_thresh[j], width, label='Best Thresholding Method', color = 'lightcoral')
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Precision Score', fontsize = font_size)
    ax.set_xlabel('Image Number', fontsize = font_size)
    ax.set_title('Precision per Image', fontsize = font_size+12)
    ax.set_xticks(x, labels, fontsize = font_size)
    ax.legend(fontsize = font_size - 2, loc = 'upper right')
    ax.set_ylim([0,1.19])
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    
    labels = ["Machine Learning", "Thresholding"]
    
    x = np.arange(len(labels)) 
    
    width = 0.25  # the width of the bars
    
    font_size = 16
    
    ax = fig.add_subplot(grid[row_size*0:row_size*1, (col_size*1)+1:(col_size*2)+1])
    
    rects1 = ax.bar(x[0], np.sum(precisions_ml[j])/len(precisions_ml[j]), width, label='Deep Learning', color = 'cornflowerblue')
    rects2 = ax.bar(x[1], np.sum(precisions_thresh[j])/len(precisions_thresh[j]), width, label='Best Thresholding Method', color = 'lightcoral')
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Precision Score', fontsize = font_size)
    ax.set_xlabel('Segmentation Method', fontsize = font_size)
    ax.set_title('Precision Average', fontsize = font_size+12)
    ax.set_xticks(x, labels, fontsize = font_size)
    ax.legend(fontsize = font_size - 2, loc = 'upper right')
    ax.set_ylim([0,1.19])
    ax.set_xlim([-0.5,1.5])
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.bar_label(rects1, [str(np.round(np.sum(precisions_ml[j])/len(precisions_ml[j]),2))], padding=0, fontsize = font_size)
    ax.bar_label(rects2, [str(np.round(np.sum(precisions_thresh[j])/len(precisions_thresh[j]),2))], padding=0, fontsize = font_size)
    
    labels = np.arange(1,len(recalls_ml[j])+1, 1).tolist()
    
    labels = [str(i) for i in labels]
    
    x = np.arange(len(labels))  # the label locations
    
    ax = fig.add_subplot(grid[(row_size*1)+1:(row_size*2)+1, col_size*0:col_size*1])
    
    rects1 = ax.bar(x - width/2, recalls_ml[j], width, label='Deep Learning', color = 'cornflowerblue')
    rects2 = ax.bar(x + width/2, recalls_thresh[j], width, label='Best Thresholding Method', color = 'lightcoral')
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Recall Score', fontsize = font_size)
    ax.set_xlabel('Image Number', fontsize = font_size)
    ax.set_title('Recall per Image', fontsize = font_size+12)
    ax.set_xticks(x, labels, fontsize = font_size)
    ax.legend(fontsize = font_size - 2, loc = 'upper right')
    ax.set_ylim([0,1.19])
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    
    labels = ["Machine Learning", "Thresholding"]
    
    x = np.arange(len(labels)) 
    
    width = 0.25  # the width of the bars
    
    font_size = 16
    
    ax = fig.add_subplot(grid[(row_size*1)+1:(row_size*2)+1, (col_size*1)+1:(col_size*2)+1])
    
    rects1 = ax.bar(x[0], np.sum(recalls_ml[j])/len(recalls_ml[j]), width, label='Deep Learning', color = 'cornflowerblue')
    rects2 = ax.bar(x[1], np.sum(recalls_thresh[j])/len(recalls_thresh[j]), width, label='Best Thresholding Method', color = 'lightcoral')
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Recall Score', fontsize = font_size)
    ax.set_xlabel('Segmentation Method', fontsize = font_size)
    ax.set_title('Recall Average', fontsize = font_size+12)
    ax.set_xticks(x, labels, fontsize = font_size)
    ax.legend(fontsize = font_size - 2, loc = 'upper right')
    ax.set_ylim([0,1.19])
    ax.set_xlim([-0.5,1.5])
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.bar_label(rects1, [str(np.round(np.sum(recalls_ml[j])/len(recalls_ml[j]),2))], padding=0, fontsize = font_size)
    ax.bar_label(rects2, [str(np.round(np.sum(recalls_thresh[j])/len(recalls_thresh[j]),2))], padding=0, fontsize = font_size)
    
    
    labels = np.arange(1,len(f1_score_ml[j])+1, 1).tolist()
    
    labels = [str(i) for i in labels]
    
    x = np.arange(len(labels))  # the label locations
    
    ax = fig.add_subplot(grid[(row_size*2)+2:(row_size*3)+2, col_size*0:col_size*1])
    
    rects1 = ax.bar(x - width/2, f1_score_ml[j], width, label='Deep Learning', color = 'cornflowerblue')
    rects2 = ax.bar(x + width/2, f1_score_thresh[j], width, label='Best Thresholding Method', color = 'lightcoral')
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('F1 Score', fontsize = font_size)
    ax.set_xlabel('Image Number', fontsize = font_size)
    ax.set_title('F1 Score per Image', fontsize = font_size+12)
    ax.set_xticks(x, labels, fontsize = font_size)
    ax.legend(fontsize = font_size - 2, loc = 'upper right')
    ax.set_ylim([0,1.19])
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    
    labels = ["Machine Learning", "Thresholding"]
    
    x = np.arange(len(labels)) 
    
    width = 0.25  # the width of the bars
    
    font_size = 16
    
    ax = fig.add_subplot(grid[(row_size*2)+2:(row_size*3)+2, (col_size*1)+1:(col_size*2)+1])
    
    rects1 = ax.bar(x[0], np.sum(f1_score_ml[j])/len(f1_score_ml[j]), width, label='Deep Learning', color = 'cornflowerblue')
    rects2 = ax.bar(x[1], np.sum(f1_score_thresh[j])/len(f1_score_thresh[j]), width, label='Best Thresholding Method', color = 'lightcoral')
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('F1 Score', fontsize = font_size)
    ax.set_xlabel('Segmentation Method', fontsize = font_size)
    ax.set_title('F1 Score Average', fontsize = font_size+12)
    ax.set_xticks(x, labels, fontsize = font_size)
    ax.legend(fontsize = font_size - 2, loc = 'upper right')
    ax.set_ylim([0,1.19])
    ax.set_xlim([-0.5,1.5])
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.bar_label(rects1, [str(np.round(np.sum(f1_score_ml[j])/len(f1_score_ml[j]),2))], padding=0, fontsize = font_size)
    ax.bar_label(rects2, [str(np.round(np.sum(f1_score_thresh[j])/len(f1_score_thresh[j]),2))], padding=0, fontsize = font_size)
    
    if save == True:
        
        plt.savefig("algo_"+str(j)+"_analysis.pdf", dpi=1000)
    
dm4_path = "data/2023_02_MicroscopyData"

data_paths = []

for folders in os.listdir(dm4_path):
    
    if 'mag' in folders:
        
        for data in os.listdir(os.path.join(dm4_path, folders)):
            
            data_paths.append(os.path.join(dm4_path, folders, data))
            
path_ml_curves = "data/deep_learning_analysis_data"

pr_curves_dim_1_algo_1 = []

pr_curves_dim_2_algo_1 = []

pr_curves_dim_1_algo_2 = []

pr_curves_dim_2_algo_2 = []

pr_curves_dim_1_algo_3 = []

pr_curves_dim_2_algo_3 = []

roc_curves_dim_1_algo_1 = []

roc_curves_dim_2_algo_1 = []

roc_curves_dim_1_algo_2 = []

roc_curves_dim_2_algo_2 = []

roc_curves_dim_1_algo_3 = []

roc_curves_dim_2_algo_3 = []

threshes = []

for data in tqdm(os.listdir(path_ml_curves)):
    
    if ("roc_curve" not in data) and ("pr_curve" not in data):
        continue
    
    if ("roc_curve" in data):
        
        for val in data_paths:
            
            if data[:-4] in val:
        
                dm3_file = dm.dmReader(val) #read the dm3 file
                break
        
        columns = ["False Positive Rate", "True Positive Rate", "Thresholds"]
        
        df = pd.read_csv(os.path.join(path_ml, data), usecols=columns)
        
        x = df["False Positive Rate"].values
        
        y = df["True Positive Rate"].values
        
        threshes.append(df["Thresholds"].values)
        
for i in range(len(threshes)-1):
    
    if i == 0:
        
        intersection = list(set(threshes[i]) & set(threshes[i+1]))
    else:
        intersection = list(set(intersection) & set(threshes[i+1]))

intersection_roc = np.round(intersection,2)

intersection_roc = np.sort(intersection_roc)

for data in tqdm(os.listdir(path_ml_curves)):

    if ("pr_curve" in data):

        for val in data_paths:

            if data[:-4] in val:

                dm3_file = dm.dmReader(val) #read the dm3 file
                break

        columns = ["Precision", "Recall", "Thresholds"]
    
        df = pd.read_csv(os.path.join(path_ml_curves, data), usecols=columns)
    
        x = df["Precision"].values
        
        y = df["Recall"].values
        
        threshes.append(df["Thresholds"].values)
    
for i in range(len(threshes)-1):
    
    if i == 0:
        
        intersection = list(set(threshes[i]) & set(threshes[i+1]))
    else:
        intersection = list(set(intersection) & set(threshes[i+1]))

intersection_pr = np.round(intersection,2)

intersection_pr = np.sort(intersection_pr)

for data in tqdm(os.listdir(path_ml_curves)):
    
    if ("roc_curve" not in data) and ("pr_curve" not in data):
        continue
    
    if ("roc_curve" in data):
        
        for val in data_paths:
            
            if data[:-4] in val:
        
                dm3_file = dm.dmReader(val) #read the dm3 file
                break
        
        resolution = get_resolution(dm3_file['pixelSize'][0],dm3_file['pixelUnit'][0])
        
        columns = ["False Positive Rate", "True Positive Rate", "Thresholds"]
        
        df = pd.read_csv(os.path.join(path_ml, data), usecols=columns)
        
        x = df["False Positive Rate"].values
        
        y = df["True Positive Rate"].values
        
        thresholds = np.round(df["Thresholds"].values,2)
        
        indices = []
        
        for ind in intersection_roc:
            
            indices.append((thresholds.tolist()).index(ind))
        
        if ('0012' in data) or ('0013' in data) or ('0015' in data): #um
        
            roc_curves_dim_1_algo_2.append(x[indices])
            
            roc_curves_dim_2_algo_2.append(y[indices])
            
        elif ('0005' in data) or ('0006' in data) or ('0008' in data): #um
        
            roc_curves_dim_1_algo_1.append(x[indices])
            
            roc_curves_dim_2_algo_1.append(y[indices])
            
        elif ('0017' in data) or ('0018' in data) or ('0019' in data): #um
        
            roc_curves_dim_1_algo_3.append(x[indices])
           
            roc_curves_dim_2_algo_3.append(y[indices])
           
    if ("pr_curve" in data):
        
        for val in data_paths:
            
            if data[:-4] in val:
        
                dm3_file = dm.dmReader(val) #read the dm3 file
                break
            
        resolution = get_resolution(dm3_file['pixelSize'][0],dm3_file['pixelUnit'][0])
        
        columns = ["Precision", "Recall", 'Thresholds']
        
        df = pd.read_csv(os.path.join(path_ml, data), usecols=columns)
        
        x = df["Precision"].values
        
        y = df["Recall"].values
        
        thresholds = np.round(df["Thresholds"].values,2)
        
        indices = []
        
        for ind in intersection_pr:
            
            indices.append((thresholds.tolist()).index(ind))
            
        if ('0012' in data) or ('0013' in data) or ('0015' in data): #um
        
            pr_curves_dim_1_algo_2.append(x[indices])
            
            pr_curves_dim_2_algo_2.append(y[indices])
            
        elif ('0005' in data) or ('0006' in data) or ('0008' in data): #um
        
            pr_curves_dim_1_algo_1.append(x[indices])
            
            pr_curves_dim_2_algo_1.append(y[indices])
            
        elif ('0017' in data) or ('0018' in data) or ('0019' in data): #um
        
            pr_curves_dim_1_algo_3.append(x[indices])
           
            pr_curves_dim_2_algo_3.append(y[indices])
                 
pr_avg_dim_1_algo_1 = (sum(pr_curves_dim_1_algo_1)/len(pr_curves_dim_1_algo_1)).tolist()

pr_avg_dim_2_algo_1 = (sum(pr_curves_dim_2_algo_1)/len(pr_curves_dim_2_algo_1)).tolist()

pr_avg_dim_1_algo_2 = (sum(pr_curves_dim_1_algo_2)/len(pr_curves_dim_1_algo_2)).tolist()

pr_avg_dim_2_algo_2 = (sum(pr_curves_dim_2_algo_2)/len(pr_curves_dim_2_algo_2)).tolist()

pr_avg_dim_1_algo_3 = (sum(pr_curves_dim_1_algo_3)/len(pr_curves_dim_1_algo_3)).tolist()

pr_avg_dim_2_algo_3 = (sum(pr_curves_dim_2_algo_3)/len(pr_curves_dim_2_algo_3)).tolist()

roc_avg_dim_1_algo_1 = (sum(roc_curves_dim_1_algo_1)/len(roc_curves_dim_1_algo_1)).tolist()

roc_avg_dim_2_algo_1 = (sum(roc_curves_dim_2_algo_1)/len(roc_curves_dim_2_algo_1)).tolist()

roc_avg_dim_1_algo_2 = (sum(roc_curves_dim_1_algo_2)/len(roc_curves_dim_1_algo_2)).tolist()

roc_avg_dim_2_algo_2 = (sum(roc_curves_dim_2_algo_2)/len(roc_curves_dim_2_algo_2)).tolist()

roc_avg_dim_1_algo_3 = (sum(roc_curves_dim_1_algo_3)/len(roc_curves_dim_1_algo_3)).tolist()

roc_avg_dim_2_algo_3 = (sum(roc_curves_dim_2_algo_3)/len(roc_curves_dim_2_algo_3)).tolist()

pr_avg_dim_1_algo_1.append(1.0)

pr_avg_dim_1_algo_1.insert(0,0)

pr_avg_dim_1_algo_1 = np.array(pr_avg_dim_1_algo_1)

pr_avg_dim_2_algo_1.append(0.0)

pr_avg_dim_2_algo_1.insert(0,1)

pr_avg_dim_2_algo_1 = np.array(pr_avg_dim_2_algo_1)

pr_avg_dim_1_algo_2.append(1.0)

pr_avg_dim_1_algo_2.insert(0,0)

pr_avg_dim_1_algo_2 = np.array(pr_avg_dim_1_algo_2)

pr_avg_dim_2_algo_2.append(0.0)

pr_avg_dim_2_algo_2.insert(0,1)

pr_avg_dim_2_algo_2 = np.array(pr_avg_dim_2_algo_2)

pr_avg_dim_1_algo_3.append(1.0)

pr_avg_dim_1_algo_3.insert(0,0)

pr_avg_dim_1_algo_3 = np.array(pr_avg_dim_1_algo_1)

pr_avg_dim_2_algo_3.append(0.0)

pr_avg_dim_2_algo_3.insert(0,1)

pr_avg_dim_2_algo_3 = np.array(pr_avg_dim_2_algo_1)

roc_avg_dim_1_algo_1.insert(0, 1.0)

roc_avg_dim_1_algo_1 = np.array(roc_avg_dim_1_algo_1)

roc_avg_dim_2_algo_1.insert(0, 1.0)

roc_avg_dim_2_algo_1 = np.array(roc_avg_dim_2_algo_1)

roc_avg_dim_1_algo_2.insert(0, 1.0)

roc_avg_dim_1_algo_2 = np.array(roc_avg_dim_1_algo_2)

roc_avg_dim_2_algo_2.insert(0, 1.0)

roc_avg_dim_2_algo_2 = np.array(roc_avg_dim_2_algo_2)

roc_avg_dim_1_algo_3.insert(0, 1.0)

roc_avg_dim_1_algo_3 = np.array(roc_avg_dim_1_algo_1)

roc_avg_dim_2_algo_3.insert(0, 1.0)

roc_avg_dim_2_algo_3 = np.array(roc_avg_dim_2_algo_1)

cols = 21
rows = 6

col_size = 10

row_size = 6

font_size = 16

#Create a subplot with 2 rows and 1 column
fig = plt.figure(figsize=(cols,rows))

grid = plt.GridSpec(rows,cols,hspace=1)

ax = fig.add_subplot(grid[row_size*0:row_size*1, col_size*0:col_size*1])

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('True Positive Rate', fontsize = font_size)
ax.set_xlabel('False Positive Rate', fontsize = font_size)
ax.set_title('Receiver Operating Characteristic Curve', fontsize = font_size+8)
ax.plot(roc_avg_dim_1_algo_1, roc_avg_dim_2_algo_1, "b", label = "ROC Curve")
ax.legend(fontsize = font_size - 1, loc = 'upper right')
ax.set_ylim([0,1.15])
ax.fill_between(roc_avg_dim_1_algo_1, roc_avg_dim_2_algo_1, 0, color = "b", alpha = 0.2)
ax.text(0.28, 0.5, "Area Under Curve = "+ str(np.round(np.mean(roc_avgs[0]), 2)), fontsize = font_size+4)
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)

ax = fig.add_subplot(grid[row_size*0:row_size*1, (col_size*1)+1:(col_size*2)+1])

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Precision', fontsize = font_size)
ax.set_xlabel('Recall', fontsize = font_size)
ax.set_title('Precision - Recall Curve', fontsize = font_size+8)
ax.plot(pr_avg_dim_1_algo_1, pr_avg_dim_2_algo_1, "r", label = "P-R Curve")
ax.legend(fontsize = font_size - 1, loc = 'upper right')
ax.set_ylim([0,1.15])
ax.fill_between(pr_avg_dim_1_algo_1, pr_avg_dim_2_algo_1, 0, color = "r", alpha = 0.2)
ax.text(0.28, 0.5, "Area Under Curve = "+ str(np.round(np.mean(pr_avgs[0]),2 )), fontsize = font_size+4)
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)


if save == True:
    
    plt.savefig("algo_1_curves.pdf", dpi=1000)


cols = 21
rows = 6

col_size = 10

row_size = 6

font_size = 16

#Create a subplot with 2 rows and 1 column
fig = plt.figure(figsize=(cols,rows))

grid = plt.GridSpec(rows,cols,hspace=1)

ax = fig.add_subplot(grid[row_size*0:row_size*1, col_size*0:col_size*1])

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('True Positive Rate', fontsize = font_size)
ax.set_xlabel('False Positive Rate', fontsize = font_size)
ax.set_title('Receiver Operating Characteristic Curve', fontsize = font_size+8)
ax.plot(roc_avg_dim_1_algo_2, roc_avg_dim_2_algo_2, "b", label = "ROC Curve")
ax.legend(fontsize = font_size - 1, loc = 'upper right')
ax.set_ylim([0,1.15])
ax.fill_between(roc_avg_dim_1_algo_2, roc_avg_dim_2_algo_2, 0, color = "b", alpha = 0.2)
ax.text(0.28, 0.5, "Area Under Curve = "+ str(np.round(np.mean(roc_avgs[1]), 2)), fontsize = font_size+4)
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)

ax = fig.add_subplot(grid[row_size*0:row_size*1, (col_size*1)+1:(col_size*2)+1])

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Precision', fontsize = font_size)
ax.set_xlabel('Recall', fontsize = font_size)
ax.set_title('Precision - Recall Curve', fontsize = font_size+8)
ax.plot(pr_avg_dim_1_algo_2, pr_avg_dim_2_algo_2, "r", label = "P-R Curve")
ax.legend(fontsize = font_size - 1, loc = 'upper right')
ax.set_ylim([0,1.15])
ax.fill_between(pr_avg_dim_1_algo_2, pr_avg_dim_2_algo_2, 0, color = "r", alpha = 0.2)
ax.text(0.28, 0.5, "Area Under Curve = "+ str(np.round(np.mean(pr_avgs[1]), 2)), fontsize = font_size+4)
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)


if save == True:
    
    plt.savefig("algo_2_curves.pdf", dpi=1000)

cols = 21
rows = 6

col_size = 10

row_size = 6

font_size = 16

#Create a subplot with 2 rows and 1 column
fig = plt.figure(figsize=(cols,rows))

grid = plt.GridSpec(rows,cols,hspace=1)

ax = fig.add_subplot(grid[row_size*0:row_size*1, col_size*0:col_size*1])

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('True Positive Rate', fontsize = font_size)
ax.set_xlabel('False Positive Rate', fontsize = font_size)
ax.set_title('Receiver Operating Characteristic Curve', fontsize = font_size+8)
ax.plot(roc_avg_dim_1_algo_3, roc_avg_dim_2_algo_3, "b", label = "ROC Curve")
ax.legend(fontsize = font_size - 1, loc = 'upper right')
ax.set_ylim([0,1.15])
ax.fill_between(roc_avg_dim_1_algo_3, roc_avg_dim_2_algo_3, 0, color = "b", alpha = 0.2)
ax.text(0.28, 0.5, "Area Under Curve = "+ str(np.round(np.mean(roc_avgs[2]), 2)), fontsize = font_size+4)
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)

ax = fig.add_subplot(grid[row_size*0:row_size*1, (col_size*1)+1:(col_size*2)+1])

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Precision', fontsize = font_size)
ax.set_xlabel('Recall', fontsize = font_size)
ax.set_title('Precision - Recall Curve', fontsize = font_size+8)
ax.plot(pr_avg_dim_1_algo_3, pr_avg_dim_2_algo_3, "r", label = "P-R Curve")
ax.legend(fontsize = font_size - 1, loc = 'upper right')
ax.set_ylim([0,1.15])
ax.fill_between(pr_avg_dim_1_algo_3, pr_avg_dim_2_algo_3, 0, color = "r", alpha = 0.2)
ax.text(0.28, 0.5, "Area Under Curve = "+ str(np.round(np.mean(pr_avgs[2]), 2)), fontsize = font_size+4)
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)


if save == True:
    
    plt.savefig("algo_3_curves.pdf", dpi=1000)
