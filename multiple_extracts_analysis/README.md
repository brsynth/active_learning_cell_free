# Description

This folder contains data and scripts necessary to generate the most informative points for each new lysate, ie: the points that allow the best prediction of the rest of the dataset (Figure 2). 


# Data:

Contains the whole data from this study.
data/no_controls contains data from rows B to O, without controls from lines A and P.

# 102_from_study:

Contains the data for the 102 points that were used in this study for other plates.


- ID_n_informative_points is the script that generated the points we used for the next steps (n varying between 8 and 24, Supplementary Figure 2e). Destination folder needs to exist.

 Randomly sampling combinations of experimental points, it chooses the best ones to predict the rest of the dataset. You need to ensure the destination folder for saving exists if you wish to use the script as it is.
- verify_informative_points allows retraining of the selected combinations on the whole dataset; prediction functions are identical but wrapping allows for testing only the selected file. For 20 points, it can be used either with final_1.csv.

Each folder with a lysate name contains resuls for this lysate.
