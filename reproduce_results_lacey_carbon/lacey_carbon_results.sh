#!/bin/bash

source C:/Users/ewalsh/AppData/Local/miniconda3/etc/profile.d/conda.sh

echo "Obtaining ML Results.."

conda run -n deep_learning python obtain_deep_learning_predictions_lacey_carbon.py

echo "Obtaining Threshold Results.."

conda run -n pyimagej python imagej_results_global.py

echo "Analysing Threshold Results.."

conda run -n deep_learning python lacey_carbon_threshold_analysis.py

echo "Analysing ML Results.."

conda run -n deep_learning python lacey_carbon_deep_learning_analysis.py

echo "Plotting Lacey-Carbon Data Results.."

conda run -n deep_learning python lacey_carbon_results_plots.py