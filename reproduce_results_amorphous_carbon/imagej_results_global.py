# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 09:36:03 2020

@author: Eoin.Walsh
"""
from __future__ import division

import imagej
import scyjava as sj
import ncempy.io.dm as dm
import os
import tifffile as tif
import sys
import numpy as np

def create_folder(folder):
    
    if os.path.exists(folder) == False:
        
        os.mkdir(folder)
        
global_thresholding = ["huang", "ij1", "intermodes", "isoData", "li", 
                       "maxEntropy", "mean", "minError", "minimum",
                       "moments", "otsu", "percentile", "renyiEntropy",
                       "shanbhag", "triangle", "yen"]

# initialize ImageJ
ij = imagej.init('sc.fiji:fiji', mode='interactive')

print(f"ImageJ version: {ij.getVersion()}")

os.chdir("..")

create_folder('experimental data/amorphous carbon grids/imageJ_segmentation')

path = os.path.join(os.getcwd(), 'experimental data/amorphous carbon grids/2023_02_MicroscopyData')

save_path = os.path.join(os.getcwd(), 'experimental data/amorphous carbon grids/imageJ_segmentation')

files = os.listdir(path)

results_list = []

for mag in files:
    
    mags = os.listdir(os.path.join(path, mag))

    for img in mags:

        image_dm = dm.dmReader(os.path.join(path, mag, img))
        
        image = image_dm['data']
        
        create_folder(os.path.join(save_path,img[:len(img)-4]))

        jslice = ij.py.to_java(image)

        # preprocess with edge-preserving smoothing.
        HyperSphereShape = sj.jimport("net.imglib2.algorithm.neighborhood.GeneralRectangleShape")
        smoothed = ij.op().run("create.img", jslice)
        ij.op().run("filter.median", ij.py.jargs(smoothed, image, HyperSphereShape(5,0, False)))
    
        for method in global_thresholding:
            
            try:
                print(method)
                
                # threshold to binary mask.
                mask = ij.op().run("threshold."+method, smoothed)

                python_mask = ij.py.from_java(mask)
                
                python_mask = (np.where(python_mask==1,0,1)*255).astype('uint8')

                tif.imwrite(os.path.join(save_path,img[:len(img)-4], 
                                        method+".tiff"), 
                                        python_mask)

            except:
                print("Problem with: "+img+", "+"method")
                pass
        