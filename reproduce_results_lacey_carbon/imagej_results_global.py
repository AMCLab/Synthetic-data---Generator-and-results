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

create_folder('experimental data/lacey carbon grids/imageJ_segmentation')

path = os.path.join(os.getcwd(), 'experimental data/lacey carbon grids/dm4')

save_path = os.path.join(os.getcwd(), 'experimental data/lacey carbon grids/imageJ_segmentation')

images = os.listdir(path)

results_list = []

for file in images:
    
    image = dm.dmReader(os.path.join(path, file))
    
    image = image['data']
    
    create_folder(os.path.join(save_path,file[:len(file)-4]))

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

            tif.imwrite(os.path.join(save_path,file[:len(file)-4], 
                                      method+".tiff"), 
                                      python_mask)

        except:
            print("Problem with: "+file+", "+"method")
            pass
        