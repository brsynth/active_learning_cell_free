# Description

This folder contains data and scripts necessary to generate the most informative points for the global dataset, ie: the points that allow the best prediction of the rest of the dataset.


# Data:

Contains the whole data from this study.
data/no_controls contains data from rows B to O, without controls from lines A and P.

# 102_from_study:

Contains the data for the 102 points that were used in this study for other plates.

# 102_most_informative is empty.

It is destined to contained results from ID_102_informative_points script.
ID_102_informative_points is the script that generated the 102 points we used for the next steps. Randomly sampling combinations of experimental points, it chooses the best ones to predict the rest of the dataset.

# 20 most informative

The folder contains data used during this study (original_20_points.csv).
It is also detsined to contain results from verify_informative_points.

# Total model on 102.
Out of curiosity, we wanted to know how our fully trained model was now able to preidct those 102 points.

# Scripts:

- ID_102_informative_points is the script that generated the 102 points we used for the next steps. Randomly sampling combinations of experimental points, it chooses the best ones to predict the rest of the dataset.
- verify_informative_points allows retraining of the selected combinations on the whole dataset; prediction functions are identical but wrapping allows for testing only the selected file. It can be used either with 20_most_informative/original_20_points.csv or 102_from_study/original_102_points.csv.


