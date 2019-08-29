# 09/08/19

Aim: analyse controls data to obtain results normalised with the same yield (global yield in the paper).

Each extract should be subtracted its own zero, whereas reference should be substracted the its zero.

The formula to use therefore becomes: 
yield = (GFP - autofluo_local)/(ref_ORI - autofluo_ORI).
We can then use the normalised yields from the control plates to normalise other plates.

# Scripts:

Each folder contains the following scripts:
- Extract_yield_own_autofluo: to extract the yield values compared to the current plate, with the correct autofluorescence subtracted.
- extracting_duplicates: taking a file containing data from 2 plates or more, allows for comparing the duplciates from those plates. See any compare_name for example.
- analysing_duplicates: calculates R2 of the 'raw yield': normalised each from its own plate, and regressions.

Each concerned folder contains normalised data.

The reference to ORI is in data_plate_ORI and the reference to PS in in data_plate_PS. Data is copy paster from those plates to the comparison plate in other datasets to normalise to those plates.


### Analyse AB:

The normalisation we proceed to is:
y_norm_to_ORI = a * y_norm_to_AB + b
normalised_to_ORI = 0.72 * normalised_to_AB + 0.0

### Analyse DH5alpha

The normalisation we proceed to is:
y_norm_to_ORI = a * y_norm_to_DH5 + b
normalised_to_ORI = 1.54 * normalised_to_DH5 + 0.38


### Analyse PS:

2 different plates can be compared from 'raw' data: the full PS plate and the PS plate used for controls are both normalised to PS. 
It is interesting to note that the controls are very well conserved: R2 of 0.9378
y_norm_to_ORI = a * y_norm_to_PS + b

The coefficients are slightly different for the full plate and the control plate.
Full: normalised_to_ORI = 1.29 * normalised_to_PS_full + 0.08
Control plate: normalised_to_ORI = 1.19 * normalised_to_PS_ctrl + 0.00

### Analyse rifaxamin:

rifaxamin_to_PS = 0.07 * rifaxamin_to_rifaxamin + 0.02
rifaxamin_to_ORI = 1.19 * rifaxamin_to_PS + 0.00

### Analyse spectinomycin: 

Remark: regression R2 is the lowest of the current workflow at 0.87

spectinomycin_to_PS = 0.15 * spectinomycin_to_spectinomycin + 0.03
spectinomycin_to_ORI = 1.19 * spectinomycin_to_PS + 0.00

