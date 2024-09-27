# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 13:25:08 2023

@author: Eoin.Walsh
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from scipy.spatial import Voronoi
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import pyfastnoisesimd as fns
import time
from tqdm import tqdm
import tifffile as tif
import math

def create_training_dirs():

    create_dir('synthetic_data')

    os.chdir("synthetic_data")

    create_dir('training_data')

    os.chdir("training_data")

    create_dir('msks')

    os.chdir("msks")

    create_dir('masks')

    os.chdir("..")

    create_dir('imgs')
    
    os.chdir('imgs')

    create_dir('images')

    os.chdir("../..")

    create_dir('validation_data')

    os.chdir("validation_data")

    create_dir('msks')

    os.chdir("msks")

    create_dir('masks')

    os.chdir("..")

    create_dir('imgs')
    
    os.chdir('imgs')

    create_dir('images')
    
    os.chdir("../../..")

def create_dir(dir):
    
    ## if the path doesn't exist, make it
    if os.path.isdir(dir) != True:
        
        os.mkdir(dir) # make the path

def perlin_new(row,col,scale,octaves,persistence,lacunarity,seed):

    shape = [row,col]
    
    N_threads = 1
    
    perlin = fns.Noise(seed=seed, numWorkers=N_threads)
    perlin.frequency = scale
    perlin.noiseType = fns.NoiseType.SimplexFractal
    perlin.FractalType = 0
    perlin.fractal.octaves = octaves
    perlin.fractal.gain = persistence
    perlin.fractal.lacunarity = lacunarity
    result = perlin.genAsGrid(shape)
  
    return result


def normalise(array):
    #change the range of an array for all values so they lie between 0 & 1. 
    array_norm = ((array - np.min(array)) / ((np.max(array)) - np.min(array))).astype("float32")
    
    return array_norm

def vorarr(regions, vertices, width, height, points,vor, dpi=100):
    fig = plt.Figure(figsize=(width/dpi, height/dpi), dpi=dpi)
    canvas = FigureCanvas(fig)
    
    ax = fig.add_axes([0,0,1,1])
    
    # colorize
    for region in regions:
        polygon = vertices[region]
        ax.plot(*zip(*polygon),'k')

    ax.plot(points[:,0], points[:,1],'.b',markerfacecolor="None",markeredgecolor='None')
    ax.set_xlim(vor.min_bound[0], vor.max_bound[0])
    ax.set_ylim(vor.min_bound[1], vor.max_bound[1])

    canvas.draw()
    return np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

def lacey_carbon_grid():

    # get random points
    np.random.seed()
    
    points = np.random.rand(200, 2)
    
    # compute Voronoi tesselation
    vor = Voronoi(points)
    
    # voronoi_finite_polygons_2d function from https://stackoverflow.com/a/20678647/425458
    regions, vertices = voronoi_finite_polygons_2d(vor)
    
    # convert plotting data to numpy array
    arr = vorarr(regions, vertices, width=3048, height=3048,points=points,vor=vor)
    
    arr = arr[500:2548,500:2548]
            
    arr = normalise(arr)
    
    arr = rgb2gray(arr)
    
    arr = np.where(arr>0.5,1,0)
    
    arr = cv2.bitwise_not(arr).astype('uint8')

    arr = np.where(arr==254,0,arr)

    #perform dilations and guassian blurring on the voronoi grid
    
    kernel_size = 29

    kernel_centre = int(kernel_size / 2)

    kernel = np.zeros((kernel_size, kernel_size), np.uint8)

    kernel = cv2.circle(kernel, (kernel_centre, kernel_centre), int(kernel_size/2) - 1, 1, -1)
    
    dilation = cv2.dilate(arr,kernel,iterations = 1)

    blur = cv2.GaussianBlur(dilation, (101,101), 21)
    
    blur1 = np.where(blur>120,255,0)
    
    final_template = normalise(blur1)
    
    points[:,0] = np.round(points[:,0]*3048)
    
    points[:,1] = np.round(np.abs(points[:,1]-1)*3048)
    
    points = remove_values(points) - 500
    
    return final_template,points

def change_brightness_lacey_carbon(array,mask):
    
   square_size = 1024
   
   rand_intx = np.random.randint(0,square_size)
   
   rand_inty = np.random.randint(0,square_size)
   
   dimming = np.random.uniform(0.5, 0.9)
   
   dimming_array = array[rand_intx:rand_intx+square_size,rand_inty:rand_inty+square_size]
   
   dimming_mask = mask[rand_intx:rand_intx+square_size,rand_inty:rand_inty+square_size]
   
   dimming_array[dimming_mask==1] = dimming_array[dimming_mask==1]*dimming
   
   array[rand_intx:rand_intx+square_size,rand_inty:rand_inty+square_size] = dimming_array

   return array

def remove_values(array):
    
    ind = np.where(array[:,0]>2548)
    
    array = np.delete(array[:], ind,axis=0)
    
    ind = np.where(array[:,1]>2548)
    
    array = np.delete(array[:], ind,axis=0)
    
    ind = np.where(array[:,0]<500)
    
    array = np.delete(array[:], ind,axis=0)
    
    ind = np.where(array[:,1]<500)
    
    array = np.delete(array[:], ind,axis=0)
    
    return array


def define_shadow_sobel3(array,rotate):
    
    np.random.seed()
    
    vertical_filter = cv2.Sobel(array, cv2.CV_32F, 1, 0)
    
    horizontal_filter = cv2.Sobel(array, cv2.CV_32F, 0, 1)
    
    orientation = np.arctan2(horizontal_filter,vertical_filter)*(180/np.pi)

    mask = np.zeros((array.shape[0],array.shape[1]))
    
    mask[vertical_filter != 0] = 1
    
    mask[horizontal_filter != 0] = 1
    
    minimum = np.min(orientation)
    
    orientation[mask != 0] = orientation[mask != 0] - minimum
    
    orientation[mask==1] = orientation[mask==1] + rotate
    
    orientation[orientation>360] = orientation[orientation>360] - 360
    
    orientation[mask != 0] = orientation[mask != 0] + minimum
    
    orientation = np.abs(orientation)
    
    final_array = normalise(orientation)
    
    final_array = (final_array - 0.5)/0.5
    
    final_array[mask!=1] = 0
    
    blur = cv2.GaussianBlur(final_array, (21,21), 2)
    
    blur_mask = np.where(blur==0,1,0)
    
    blur = normalise(blur)
    
    blur = ((blur - 0.5)/0.5)*3

    return blur

def normalise_dataset(path):
    
    imgs = os.listdir(path)

    maximum = 0

    minimum = 256

    for i in tqdm(range(len(imgs))):
        
        img_path = os.path.join(path,imgs[i]) #image path
        
        img = tif.imread(img_path) #read in the image
        
        max_val = np.max(img) #max pixel value of image
        
        min_val = np.min(img) #min pixel value of image
        
        #if max pixel value less than minimum for all images so far, let 
        #max pixel value be the overall minimum
        if max_val>maximum: 
            
            maximum = max_val
        
        #if min pixel value less than minimum for all images so far, let 
        #min pixel value be the overall minimum
        if min_val<minimum:
            
            minimum = min_val
    
    #normalise all of the image data based on the max and min found for the
    #overall dataset
    for i in tqdm(range(len(imgs))):
        
        img_path = os.path.join(path,imgs[i])
        
        img = tif.imread(img_path)
        
        img2 = ((img-minimum)/(maximum-minimum)).astype('float32')
        
        img2 = (img2*((2**16) - 1)).astype('uint16')
        
        tif.imwrite(img_path,img2)
        
    return maximum,minimum
        
def define_particle_shadow_sobel3(array,rotate):
    
    np.random.seed()
    
    vertical_filter = cv2.Sobel(array, cv2.CV_32F, 1, 0)
    
    horizontal_filter = cv2.Sobel(array, cv2.CV_32F, 0, 1)
    
    orientation = np.arctan2(horizontal_filter,vertical_filter)*(180/np.pi)
    
    mask = np.zeros((array.shape[0],array.shape[1]))
    
    mask[vertical_filter != 0] = 1
    
    mask[horizontal_filter != 0] = 1
    
    minimum = np.min(orientation)
    
    orientation[mask != 0] = orientation[mask != 0] - minimum
    
    orientation[mask==1] = orientation[mask==1] + rotate
    
    orientation[orientation>360] = orientation[orientation>360] - 360  
    
    orientation[mask != 0] = orientation[mask != 0] + minimum
    
    orientation = np.abs(orientation)
    
    final_array = normalise(orientation)
    
    final_array = (final_array - 0.5)/0.5
    
    final_array[mask!=1] = 0
    
    blur = cv2.GaussianBlur(final_array, (21,21), 2)
    
    if (np.abs(np.min(blur))/(np.max(blur)-np.min(blur))) > 0.55:
        
        blur[blur < -1*(0.549*(np.max(blur))/(0.451))] = -1*(0.549*(np.max(blur))/(0.451))
        
    elif (np.abs(np.min(blur))/(np.max(blur)-np.min(blur))) < 0.45:
        
        blur_mask = np.where(blur == 0, 1, 0)
        
        interval_min = -0.451
        interval_max = np.max(blur)
        
        blur = ((blur - np.min(blur)) / (np.max(blur) - np.min(blur))) * (interval_max - interval_min) + interval_min
        
        blur[blur_mask==1] = 0
        
    blur = normalise(blur)
    
    blur = np.abs(((blur - 0.5)/0.5)*2)
    
    mask_final = np.where(blur>0.2,1,0)
    
    mask_final = np.where((array+mask_final)>0,1,0)
    
    return blur

def scale_vals(array):
    
    new_max = 2.7
    
    new_min = -3.2
    
    new_array = ((new_max-new_min)*((array-np.min(array))/(np.max(array)-np.min(array)))) + new_min
    
    return new_array

def remove_list_elements(array):
    
    index_1 = [coordinate[0] for coordinate in array]
    
    remove_index = [index for index, val in enumerate(index_1) if (val > 2020 or val < 25)]
    
    index_2 = [coordinate[1] for coordinate in array]
    
    remove_index = list(set(remove_index + [index for index, val in enumerate(index_2) if (val > 2020 or val < 25)]))
    
    array = np.delete(array,remove_index,axis=0)
    
    return array

def remove_points_outside_range(array,lim):
    
    index_1 = [coordinate[0] for coordinate in array]
    
    remove_index = [index for index, val in enumerate(index_1) if (val > lim-30 or val < 30)]
    
    index_2 = [coordinate[1] for coordinate in array]
    
    remove_index = list(set(remove_index + [index for index, val in enumerate(index_2) if (val > lim-30 or val < 30)]))
    
    array = np.delete(array,remove_index,axis=0)
    
    return array

def remove_points_one_areas(array,points):
    
    non_one_points = []
    
    for point in points:
        
        if array[int(point[0]):int(point[0]+1),int(point[1]):int(point[1]+1)] == 0:
           
          non_one_points.append(list(point))
          
    return np.array(non_one_points)


def image_rotation(image,angle):
    
    row,col = image.shape # assign row and column length values based on array dimensions
    
    n = (row/np.sqrt(2))*math.sin(angle*math.pi/180) # length of the side of the rotated section we want to cut out
    
    pts1 = np.float32([[n,0],[row,n],[0,col-n],[row-n,col]]) # coordinates that you want to Perspective Transform
    
    pts2 = np.float32([[0,0],[row,0],[0,col],[row,col]]) # Size of the Transformed Image (original array size)

    M = cv2.getPerspectiveTransform(pts1,pts2) # computer transformation matrix
    
    result = cv2.warpPerspective(image,M,(row,col)) # apply transformation matrix to the original image
    
    return result

def gradient_brightness(array):
    
    random = np.random.uniform(0,1) # generate random number.
    
    row,col = array.shape[0],array.shape[1] # assign values to row and col variables, which are the shape of the input array.
    
    rand_intx = 0 # instantiate rand integer variable along x axis.
   
    rand_inty = 0 # instantiate rand integer variable along y axis.
    
    if random > 0.25 and random < 0.75:
        
        brighten = np.random.uniform(0.5,0.8)
            
        segment = np.arange(0,brighten,brighten/row,dtype='float32')
        
    else:
        
        dimming = np.random.uniform(0.2,0.5) # random number which dictates how much we dim the array by, lower = darker.
    
        # row vector equal in length to array dimension, values of array spread evenly between dimming variable and 1.
        # This will be used to give the gradient fall off in brightness in the final array.
        segment = np.arange(dimming,1,(1-dimming)/row,dtype='float32')
    
    if len(segment) > row:
        
        segment = segment[:row].reshape(row,1)
        
    else:
        
        segment = segment.reshape(row,1)
    
    segment = np.tile(segment,col) # turn segment (500,1) vector into repeating (500,500) verctor, so that we account for all array columns
    
    if random > 0.5: # if random number greater than 0.5, flip the array so dimming happens in opposite direction
        
        segment = np.flip(segment) # flip the array for blurring in the opposute direction
    
    if random > 0.25 and random < 0.75:
        
         array = (array+segment) # increase the brightness by adding array and segment
         
         array = array/np.max(array) # normalise
         
    else:
        
         array = np.multiply(segment,array) # decrease brightness by multiplying array and segment
         
         array = array/np.max(array) # normalise
    
    return array

def create_gradient_particle_ellipse(points, number, particle_intensity_range, gradient=True,shadow=True):

    points = remove_list_elements(points)
    
    no_of_points = np.linspace(0,len(points)-1,len(points))
    
    if number > len(no_of_points):
        
        number = len(no_of_points)
        
    index = np.random.choice(no_of_points,size=number,replace = False).astype('uint8')
    
    plot_points = points[index]
    
    size = 300 # size of the array
    
    img_array = np.zeros((2048,2048)) # instantiate the image array
    
    mask_array = np.zeros((2048,2048)) # instantiate the mask array
    
    for point in plot_points:
        
        centre = (int(size/2),int(size/2))
        
        gradient_particle_array = np.zeros((size,size)) # empty array for particle color gradient
        
        particle_array = np.zeros((size,size)) # empty array for particle color
        
        ellipse_array = np.zeros((size,size)) # empty array to draw particle circle in
        
        axes_length = (np.random.randint(100,150),np.random.randint(100,150)) # length of the major and minor axis of ellipse
        
        angle = np.random.randint(0,180) # angle ellipse rotates through
        
        start = 0 # proportion of ellipse fill in, start angle
        
        end = 360 # proportion of ellipse filled in, end angle
        
        top_left_x = np.random.randint(0,2048-int(50))#(1/magnification))) # x-coordinate for pasting particle
        
        top_left_y = np.random.randint(0,2048-int(50))#(1/magnification))) # y-coordinate for pasting particle
            
         ## draw circle in the array
        ellipse_array = (cv2.ellipse(ellipse_array,centre,axes_length,angle,start,end,(255,255,255),-1)/255).astype('uint8') 
         
        noise_array = np.random.normal(0.5,0.2,(size,size)) # fill array with Gaussian noise
        
        noise_array = normalise(noise_array) # normalise the array
        
        if gradient == True:
                
            gradient_array = gradient_brightness(noise_array) # create gradient change in brightness for array
            
            gradient_array = normalise(gradient_array) # normalise the array
        
            rotated_array = image_rotation(gradient_array,np.random.randint(0,180)) # rotate the array
            
            rotated_array = normalise(rotated_array) # normalise array
        
            gradient_particle_array[ellipse_array==1] = rotated_array[ellipse_array==1] # final gradient for particle
        
        else:
            gradient_array = np.ones_like(noise_array)
            
            gradient_particle_array[ellipse_array==1] = gradient_array[ellipse_array==1] # final gradient for particle
        
        particle_darkness = np.random.uniform(particle_intensity_range[0], 
                                              particle_intensity_range[1]) # particle color pixel values
       
         ## coloring in the particle with random uniform noise
        particle_array[ellipse_array==1] = np.random.uniform(particle_darkness,particle_darkness+0.5,len(ellipse_array[ellipse_array==1]))
         
         ## resize the particle array
        final_particle_array = cv2.resize(particle_array*gradient_particle_array,
                                          (int(50),int(50)),
                                          interpolation=cv2.INTER_NEAREST)
         
         ## resize the particle mask array
        ellipse_array = cv2.resize(ellipse_array,
                                   (int(50),int(50)),
                                   interpolation=cv2.INTER_NEAREST)
        
        if shadow == True:
            
            shadow_array= define_particle_shadow_sobel3(ellipse_array, np.random.randint(0,180)) # define particle shadow
         
            final_particle_array = final_particle_array + shadow_array
            
         ## place particle in the overall image array
        img_array = place_particles_array(img_array,final_particle_array,int(point[0]),int(point[1]))
     
         ## place particle mask in the overall mask array
        mask_array = place_particles_array(mask_array,ellipse_array,int(point[0]),int(point[1]))
        
    return img_array,mask_array

def place_particles_array(array,particle,x_coordinate,y_coordinate):

    try:
        
        ## using indexing, place particle in the image array
        array[x_coordinate-int(particle.shape[0]/2):x_coordinate-int(particle.shape[0]/2)+particle.shape[0],
            y_coordinate-int(particle.shape[1]/2):y_coordinate-int(particle.shape[1]/2)+particle.shape[1]][particle != 0] = particle[particle != 0]
    except:
        pass
    
    return array

def create_dirs(directory):
    
    if os.path.exists(os.path.join(directory, 'training_data')) == False:
        
        os.mkdir(os.path.join(directory, 'training_data'))
        
        os.mkdir(os.path.join(directory, 'training_data', 'images'))
        
        os.mkdir(os.path.join(directory, 'training_data', 'masks'))
        
        os.mkdir(os.path.join(directory, 'training_data', 'images', 'imgs'))
        
        os.mkdir(os.path.join(directory, 'training_data', 'masks', 'msks'))
        
    if os.path.exists(os.path.join(directory, 'validation_data')) == False:
        
        os.mkdir(os.path.join(directory, 'validation_data'))
        
        os.mkdir(os.path.join(directory, 'validation_data', 'images'))
        
        os.mkdir(os.path.join(directory, 'validation_data', 'masks'))
        
        os.mkdir(os.path.join(directory, 'validation_data', 'images', 'imgs'))
        
        os.mkdir(os.path.join(directory, 'validation_data', 'masks', 'msks'))
        
def generate_data(number_image_batches, 
                  number_particles,
                  magnification_range,
                  particle_intensity_range, 
                  background_brightness_range,
                  graphene_intensity_range,
                  lacey_carbon_intensity_mean_range,
                  lacey_carbon_intensity_std,
                  random_lacey_carbon_brightness_change,
                  gradient=False, 
                  shadow=False, 
                  rand_mag = True,
                  include_sobel_graphene = True,
                  graphene_intensity_mean_rand = True,
                  folder='training_data'):
    
    baseline_mag = 4300
    
    count = 0
    
    np.random.seed()
    
    if rand_mag == True:
        
        iterant = 1
        
        number_image_batches *= 3
    
    else:
        iterant = 3
        
    for j in tqdm(range(number_image_batches)):
        
        for i in range(iterant):
            
            if rand_mag == True:
                
                mag = baseline_mag/(np.random.randint(magnification_range[0], magnification_range[1]))
            else:
                mag_mult = ((magnification_range[1] - magnification_range[0])/2)*i
                
                mag = baseline_mag/(magnification_range[0] + mag_mult)
                
            vor_array,points = lacey_carbon_grid()
            
            if include_sobel_graphene == True:
                
                shad_arr = define_shadow_sobel3(vor_array,np.random.randint(0,180))
            
            #copy_original_array
            graphene_carbon_location = vor_array.copy() 
            
            graphene_carbon_location = cv2.resize(graphene_carbon_location[:int(2048*mag),:int(2048*mag)],
                                                    dsize=(2048,2048),
                                                    interpolation=cv2.INTER_NEAREST)
            if include_sobel_graphene == True:
            
                shad_arr = cv2.resize(shad_arr[:int(2048*mag),:int(2048*mag)],
                                                        dsize=(2048,2048),
                                                        interpolation=cv2.INTER_NEAREST)
            
            points = remove_points_outside_range(points,int(2048*mag))
            
            points = ((points/int(2048*mag))*2048).astype('int')
            
            points = remove_points_one_areas(graphene_carbon_location,points)
            
            points = ((points/2048)*(int(2048*mag))).astype('int')

            if len(points) == 0:
                
                points = []
                
                for i in range(5):
                    
                    points.append([np.random.randint(0,int(2048*mag)),np.random.randint(0,int(2048*mag))])
                
                points = np.array(points)
                
                points = ((points/int(2048*mag))*2048).astype('int')
                
                points = remove_points_one_areas(graphene_carbon_location,points)
                
                points = ((points/2048)*(int(2048*mag))).astype('int')
                
                number_particles = 1
                
            ellipse_array, ellipse_array_mask = create_gradient_particle_ellipse(points,
                                                                                number_particles,
                                                                                particle_intensity_range,
                                                                                gradient,
                                                                                shadow)
            
            ellipse_array = cv2.resize(ellipse_array[:int(2048*mag),:int(2048*mag)],
                                                dsize=(2048,2048),
                                                interpolation=cv2.INTER_NEAREST)
            
            ellipse_array_mask = cv2.resize(ellipse_array_mask[:int(2048*mag),:int(2048*mag)],
                                                dsize=(2048,2048),
                                                interpolation=cv2.INTER_NEAREST)
            
            graphene_carbon_mask = graphene_carbon_location.copy() 
            
            if graphene_intensity_mean_rand == True:
                
                graphene_carbon_location[graphene_carbon_location==0] = np.random.normal(np.random.uniform(graphene_intensity_range[0],graphene_intensity_range[1]),
                                                                                         graphene_intensity_range[2],
                                                                                         len(graphene_carbon_location[graphene_carbon_location==0]))                     
            else:
                graphene_carbon_location[graphene_carbon_location==0] = np.random.normal(graphene_intensity_range[0],
                                                                                         graphene_intensity_range[1],
                                                                                         len(graphene_carbon_location[graphene_carbon_location==0]))
                
            graphene_carbon_location[graphene_carbon_location==1] = np.random.normal(np.random.uniform(lacey_carbon_intensity_mean_range[0],
                                                                                                       lacey_carbon_intensity_mean_range[1]),
                                                                                     lacey_carbon_intensity_std,
                                                                                     len(graphene_carbon_location[graphene_carbon_location==1]))
            if include_sobel_graphene == True:
            
                lacey_carb = shad_arr +  graphene_carbon_location
            
            else:
                lacey_carb = graphene_carbon_location.copy()
                
            lacey_carb = lacey_carb + ellipse_array
                                            
            #instantiate perlin noise
            perlin1 = perlin_new(row = 2048, #row size
                                col = 2048, #column size
                                scale = 2.4*mag, #frequency of perlin noise in the array
                                octaves = 6, #number of layers of noise 
                                persistence = 0.4, #how much eaxh layer is weighted against the previous layer
                                lacunarity = 2.0, #strength of the layer in question
                                seed = np.random.randint(0,200000)) #random seed for noise
            
            perlin1 = normalise(perlin1) #normalise perlin noise
            
            final_img = normalise(lacey_carb + perlin1) #add the carbon + graphene background with perlin noise
            
            final_img = normalise(final_img + np.random.normal(0.5,0.15,size=(2048,2048))) #add in the gaussian noise

            if random_lacey_carbon_brightness_change == True:
                
                final_img = change_brightness_lacey_carbon(final_img,graphene_carbon_mask)
        
            final_img = final_img + np.random.uniform(background_brightness_range[0], 
                                                      background_brightness_range[1])
            
            ellipse_array_mask = ellipse_array_mask.astype('uint8')
            
            tif.imwrite(os.path.join(os.getcwd(), 'synthetic_data', folder+'/imgs/images/image_'+str(count)+'.tif'),final_img)
            
            tif.imwrite(os.path.join(os.getcwd(), 'synthetic_data', folder+'/msks/masks/image_'+str(count)+'.tif'),ellipse_array_mask)
            
            count = count + 1

def choose_dataset(dataset_type):
    
    create_training_dirs()

    start = time.time()
   
    if dataset_type == "small mag":

        number_image_batches = 10

        number_particles = np.random.randint(15,20)

        background_brightness_range = (-0.2, 0.2)

        magnification_range = (4300, 5200)

        particle_intensity_range = (-3, -4.1)

        graphene_intensity_range = (0.5,0.1)

        lacey_carbon_intensity_mean_range = (-0.7, 0.1)

        lacey_carbon_intensity_std = 0.05

        random_lacey_carbon_brightness_change = True
        
        gradient = False
        
        shadow = False
        
        rand_mag = True
        
        graphene_intensity_mean_rand = False
        
        include_sobel_graphene = True

    elif dataset_type == "medium mag":
        
        number_image_batches = 3

        number_particles = np.random.randint(1, 4)

        background_brightness_range = (-0.2, 0.2)

        magnification_range = (14000, 30000)

        particle_intensity_range = (-12, -14)

        graphene_intensity_range = (0.8,0.1)

        lacey_carbon_intensity_mean_range = (-0.7, 0.1)

        lacey_carbon_intensity_std = 0.05

        random_lacey_carbon_brightness_change = True

        gradient = True

        shadow = True

        rand_mag = False
        
        graphene_intensity_mean_rand = False
        
        include_sobel_graphene = True


    elif dataset_type == "large mag":
        
        number_image_batches = 3

        number_particles = np.random.randint(1, 4)

        background_brightness_range = (-0.2, 0.2)

        magnification_range = (43000, 77500)

        particle_intensity_range = (-3, -4.1)

        graphene_intensity_range = (-0.9, -0.7, 0.1)

        lacey_carbon_intensity_mean_range = (-3, -2.5)

        lacey_carbon_intensity_std = 0.1

        random_lacey_carbon_brightness_change = False

        gradient = True

        shadow = True

        rand_mag = False

        graphene_intensity_mean_rand = True

        include_sobel_graphene = False


    else:
        
        raise ValueError("This type of dataset does not exist. Select from: 'small mag', 'medium mag', and 'large mag.'")
    
    if number_image_batches > 3:
        
        number_image_batches = int(number_image_batches/3)

    generate_data(number_image_batches, 
                  number_particles,
                  magnification_range,
                  particle_intensity_range, 
                  background_brightness_range,
                  graphene_intensity_range,
                  lacey_carbon_intensity_mean_range,
                  lacey_carbon_intensity_std,
                  random_lacey_carbon_brightness_change,
                  gradient=gradient, 
                  shadow=shadow, 
                  graphene_intensity_mean_rand = graphene_intensity_mean_rand,
                  include_sobel_graphene = include_sobel_graphene,
                  rand_mag = rand_mag,
                  folder='validation_data')
    
    path = os.path.join(os.getcwd(), 'synthetic_data/training_data/imgs/images')

    normalise_dataset(path)
    
    path = os.path.join(os.getcwd(), 'synthetic_data/validation_data/imgs/images')
    
    normalise_dataset(path)

    print('\nTime Taken: ',np.round(time.time()-start,2),' seconds')
    