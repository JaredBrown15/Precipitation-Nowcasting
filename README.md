Created by: Jared Brown, Wright State University, 06/19/2025

# Precipitation-Nowcasting
Precipitation Nowcasting Using an AI-Enabled Probabilistic Geometric Method

This repository contains code and resources used to complete my master's thesis at Wright State University. Data is not included in this repository and must be included separately. This repo was not used during the thesis (which would have been a GREAT IDEA), but is housing the finalized code.

You can find the thesis publication at https://etd.ohiolink.edu/acprod/odb_etd/etd/r/1501/10?clear=10&p10_accession_num=wright1748891568013252

Written in python using the SEVIR dataset. Information on SEVIR available here: https://github.com/MIT-AI-Accelerator/neurips-2020-sevir

## IMPORTANT CODE FILES
If memory serves me correctly, these are the files I used for final analysis. Files are disorganized here, but I have learned from this mistake

Thesis/Code/Finalized Code/MASTER SCRIPT - Following Design of Experiments.ipynb
Thesis/Code/Finalized Code/Spear.py
Thesis/Code/Finalized Code/master_script_helper_functions.py

## Summary of methodology
See Precipitation-Nowcasting/Thesis-Defense.pptx for additional details.

Ellipses were fit to storm radar signatures using a process similar to PCA. Radar images were binarized, then the covariance matrix was calculated. Eigenvectors/eigenvalues corresponded to ellipse axis directions/magnitudes respectively.
1st order backward differencing was used to calculate storm (ellipse) velocity. These features were then normalized and passed into an LSTM architecture to predict ellipse location, size, and speed in the future. 
