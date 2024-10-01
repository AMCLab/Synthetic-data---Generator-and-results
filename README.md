## Synthetic Silica Particle Generator - Chapter 5

### Summary

This repository contains all the scripts necessary to replicate the results presented in the synthetic data paper paper draft at https://www.overleaf.com/project/663c753ae7583a859de7f2a1

Enjoy, and any questions can be sent to eoinpadraigwalsh26+research@gmail.com.

### Instructions

- First, clone this repository to the directory of your choosing on your local machine using ```git clone```.

- Next, download the experimental data available at https://zenodo.org/uploads/11569905, and extract the data to the cloned Github repository location on your local machine.
  
## Conda/Mamba Environment Setup

- install miniconda or anaconda from https://docs.conda.io/projects/miniconda/en/latest/

- We create a conda environment and install relevant packages for replicating the deep learning results and conducting data analysis.
    1. ```conda create -n deep_learning python=3.8.19```
    2. Run ```pip install -r requirements_deep_learning.txt``` within the newly created conda environment, and in the cloned Github directory.
  
- We create a mamba environment and install relevant packages for replicating the ImageJ results.
    1. Install mamba using the instructions here: https://py.imagej.net/en/latest/Install.html
    2. Activate conda-forge ```conda config --add channels conda-forge``` and ```conda config --set channel_priority strict```.
    3. Install pyimagej and create an environment for it in conda using: ```mamba create -n pyimagej pyimagej openjdk=11```
    4. Run ```pip install -r requirements_imagej.txt``` within the newly created conda environment, and in the cloned Github directory.

## Running Code

- Run the ```all_results.sh``` shell script to automatically run all of the Python scripts required to obtain the paper results.
- The raw results data are available at ```experimental data\amorphous carbon grids``` and ```experimental data\lacey carbon grids```.
- The result plots are available in ```reproduce_results_lacey_carbon/figures``` and ```reproduce_results_amorphous_carbon/figures```.

- All individual Python scripts are available at ```reproduce_results_lacey_carbon``` and ```reproduce_results_amorphous_carbon```, with the ordering in which they should be run detailed in shell scripts in both folders.
