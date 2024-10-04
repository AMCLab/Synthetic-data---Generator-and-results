#!/bin/bash

echo "Obtaining ML Results.."

conda run -n deep_learning python gold_nanoparticle_deep_learning_predictions.py

echo "Obtaining Threshold Results.."

conda run -n pyimagej python imagej_results_global.py

echo "Analysing Threshold Results.."

conda run -n deep_learning python gold_nanoparticle_threshold_analysis.py

echo "Analysing ML Results.."

conda run -n deep_learning python gold_nanoparticle_deep_learning_analysis.py

echo "Plotting Amorphous Carbon Data Results.."

conda run -n deep_learning python gold_nanoparticles_results_plots.py