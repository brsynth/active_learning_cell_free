# Description.

The aim of this folder is to regroup all necessary scripts for generating data and models presented in Large scale active-learning-guided exploration to maximize cell-free production by Olivier Borkowski*, Mathilde Koch*, Agnès Zettor, Amir Pandi, Angelo Cardoso Batista, Paul Soudier and Jean-Loup Faulon.

# Active learning loop.

This folder contains data both for 
- generating the active learning loop (learn_and_suggest script)
- generating model statistics at each iteration (generate_model_statistics)
- initial_plate_generation: how the first plate was generated

# Compound effect analysis

Contains scripts to analyse the effect of compounds in different lysates with a linear regression or mutual information.

# ECHO handling scripts

This folder contains scripts for handling the ECHO machine, from a file of concentrations to test to the instructions to the machine to data extraction and quality control.

# Multiple extracts analysis.

This folder contains scripts to extract the most informative 20 points to predict 102 other points, as well as various analyses than were ran using those 20 points.

# Whole lysate most informative points.

The aim of this folder is to extract the most informative 102 (or 20) points that predict the full 1017 points dataset.
Functions are similar to the ones that do the same thing in other_extract analysis. The difference is the wrapping around the input concentrations to test, or data to comapre to.

# Controls analysis:

This folder contains the data and scripts used to extract absolute yields (ie: compared to lysate of origin) for all other lysates.

# Requirements:

To run those functions, required packages are:
- nb_conda_kernels (for running the jupyter notebook in the correct environment)
- jupyter_contrib_nbextensions (tools for better Jupyter notebooks): conda install -c conda-forge jupyter_contrib_nbextensions
- numpy (for array handling)
- matplotlib (for visualisation)
- scikit-learn (for machine learning). Version 0.19.1 has to be used.
