## Synthetic Lacey-Carbon Grid + Silica Particle Generator & Paper Results

### Summary

This repository contains all the scripts necessary to replicate the results presented in the synthetic data paper. It also contains instructions and code on generating synthetic lacey carbon on ultra-thin carbon grids.

NB: This repository and these instructions were developed to work using a linux OS.

### Instructions

- First, clone this repository to the directory of your choosing on your local machine using ```git clone```.

- Next, download the experimental data available at https://zenodo.org/uploads/11569905, and extract the data to the cloned Github repository location on your local machine.
  
## Conda Environment Setup

We create 3 distinct environments to run the code in this repository, one for generating the synthetic data, another for utilising the deep learning algorithms, finally an environment for obtaining global thresholding results from ImageJ.

- install miniconda or anaconda from https://docs.conda.io/projects/miniconda/en/latest/

- We create a conda environment and install relevant packages for replicating the deep learning results and conducting data analysis.
    1. ```conda create -n lacey_carbon_generator python=3.8.19```
    2. Run ```pip install -rrequirements_lacey_carbon_generator.txt``` within the newly created conda environment, and in the cloned Github directory.

- We create a conda environment and install relevant packages for replicating the deep learning results and conducting data analysis.
    1. ```conda create -n deep_learning python=3.8.19```
    2. Run ```pip install -r requirements_deep_learning.txt``` within the newly created conda environment, and in the cloned Github directory.
  
- We create another conda environment and install relevant packages for replicating the ImageJ results.
    1. Install pyimagej and create an environment for it in conda using: ```conda create -n pyimagej pyimagej openjdk=11```
    2. Run ```pip install -r requirements_imagej.txt``` within the newly created conda environment, and in the cloned Github directory.

## Running Code - RESULTS

- Navigate to ```reproduce_results_amorphous_carbon``` directory and run ```bash amorphous_carbon_results.sh```
- Navigate to ```reproduce_results_lacey_carbon``` directory and run ```bash lacey_carbon_results.sh```

- The raw results data are available at ```experimental data\amorphous carbon grids``` and ```experimental data\lacey carbon grids```.
- The result plots are available in ```reproduce_results_lacey_carbon/figures``` and ```reproduce_results_amorphous_carbon/figures```.

- All individual Python scripts are available at ```reproduce_results_lacey_carbon``` and ```reproduce_results_amorphous_carbon```, with the ordering in which they should be run detailed in shell scripts in both folders.

## Running Code - SYNTHETIC DATA GENERATION

- Run the ```generate_data.py``` script to reproduce the synthetic data generated for the paper. Further instructions on usage can be found as docstrings in the script itself.

- Experiment and augment with synthetic data parameters for your own use case using the ```synthetic_data_experimentation.py``` script.

## Running Code - ALGORITHM TRAINING

NB: You will need a GPU to run the following scripts efficiently.

- Run the ```PAT4NANO_ml_training.py``` script to train a deep learning algorithm using the generated synthetic data in the /synthetic_data directory. 

- Run the ```grid_search_PAT4NANO.py``` script to optimise hyperparameters in the deep learning algorithm for your generated synthetic datasaet.
