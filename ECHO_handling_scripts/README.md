# Description

This folder contains scripts that handle conversion of files from a file containing concentrations one wishes to try out to the instructions for the ECHO machine and the extraction of information from the TECAN (mapping values back to the concentration array).

# completed_concentration

The aim of this script is to complete a concentration file without constant values (extract) to a file that has those columns.
This file needs to be run on the 3 files: concentrations_line_A, concentrations_example, concentrations_line_P. 
This will generate the following files: concentrations_line_A_completed, concentrations_example_completed, concentrations_line_P_completed.

Modified only the cocentrations file, not the scripts for this

# concentration_to_volume

This script converts a concentration file to a volume file.
It needs to be run on concentrations_line_A_completed, concentrations_example_completed, concentrations_line_P_completed, and creates the files:
example_volumes, line_A_volumes, line_P_volumes.

Script is modified and volumes are compatible.

# Remark:
For easier pipetting later in the process, volume files can be sorted by water volume or amino acids volume so that high volumes are grouped together.

# volume_to_echo

This script converts a volume file (example) to the following files, using the 2 control files (line_A_volumes, line_P_volumes):
- example_named_volumes, line_A_named_volumes, line_P_named_volumes.
- example_instructions
- example_water (for high volumes to pipet)
- example_aa (for high volumes to pipet)

Takes only volumes, no modification required.

# named_volumes_to_concentrations

This script converts named volume files to named concentration files. It needs to be run on example_named_volumes, line_A_named_volumes, line_P_named_volumes.
And it will return example_concentrations_reconstituted, line_A_concentrations_reconstituted, line_P_concentrations_reconstituted.

Works.

Concentrations need to be merged (by order of well name, line_A at the top of the file and line P at the bottom).

# Extract yield:

Takes 2 inputs: raw results from a TECAN (example_TECAN) and reconstituted concentrations from the previous step.
Returns:
- yield_and_std
- outliers (returns outliers)
- everything: all results in the same file
- std, ratio, mean: draw the plaque with std, ratio between mean and std and mean to see whether outliers are localised on the plaque.
- comments: about outliers.

