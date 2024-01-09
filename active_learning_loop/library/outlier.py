def outliers(sample, threshold = 1):
    """
    Identify outliers in a given sample.

    Parameters:
    - sample (array-like): The input sample for which outliers are to be identified.
    - threshold (float, optional): The threshold value for determining outliers.
                                  Data points beyond this threshold are considered outliers.
                                  Default is 1.

    Returns:
    - outliers (array): An array containing the identified outliers based on the specified threshold.
                        If fewer than two outliers are found, the array will contain the outliers;
                        otherwise, the 3 points consider outliers.

    Example:
    >>> sample_data = np.array([1, 2, 3, 10, 15, 20, 100])
    >>> outliers(sample_data, threshold=2)
    array([100])
    """
    import numpy as np
    if not isinstance(sample, np.ndarray):
        sample = np.array(sample)

    mean = np.mean(sample, axis = 1)
    mean = np.tile(mean,(sample.shape[1], 1)).T

    std = np.std(sample, axis = 1)
    std = np.tile(std,(sample.shape[1], 1)).T

    z_scores = (sample-mean)/std
    outliers_z = np.abs(z_scores) > threshold
    result = []
    for i in range(len(outliers_z)):
        condi_row = outliers_z[i]
        data_row = sample[i]
        if len(data_row[condi_row]) < 2:  
            result.append(data_row[condi_row])
        else: 
            result.append(data_row)
    return result