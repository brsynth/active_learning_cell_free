import glob
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MaxAbsScaler


def import_data(folder_for_data, verbose = True):

    files = glob.glob(folder_for_data + "\\*.csv")

    # Initialize a DataFrame with NaN values
    concatenated_data = pd.DataFrame()
    size_list = []

    # Initialize a flag to check if columns are consistent across files
    consistent_columns = True

    # Iterate through files
    for file in files:
        df = pd.read_csv(file)

        # Check if columns are consistent across files
        if concatenated_data.columns.empty:
            concatenated_data = df
            size_list.append(len(df))
        elif not concatenated_data.columns.equals(df.columns):
            consistent_columns = False
            print(f"Ignoring file {file}: Column orders are not consistent.")
            files.remove(file)
        else: 
            concatenated_data = pd.concat([concatenated_data, df], ignore_index=True)
            size_list.append(len(df))

    if verbose:
        print("Read ", len(files), " files: ")
        for file in files:
            data_name = os.path.basename(file)
            print("- ", data_name)
        display(concatenated_data)

        if consistent_columns:
            print("All files have consistent column orders.")
        else:
            print("Some files have inconsistent column orders.")

    return concatenated_data, size_list


def select_from_iteration (data, selected_plate = (0,1)):
    """
    Obtain data from the desired plates, and not the whole data.
    """
    selected_plates = data[selected_plate[0]:selected_plate[1]]
    current_data = np.concatenate(selected_plates, axis = 0)
    return(current_data)


def import_parameter(parameter_file, nb_new_data_predict, sep = ';', verbose = False):

    df = pd.read_csv(parameter_file, sep = sep)
    element_list = df.iloc[:,0].to_list()
    element_max = df.iloc[:,1].to_list()
    concentration = df.iloc[:,2:].to_numpy()

    multi_max = np.repeat([element_max], concentration.shape[1], axis = 0).T
    concentration = multi_max*concentration

    row,col = concentration.shape
    combi = col**row
    if verbose:
        print(f"Number of metabolites : {len(element_list)}")
        print(f"Number of combinations - poolsize : {combi}")
        print(f"Searching ratio : {round(nb_new_data_predict*100/combi, 2)} %")
        print(f"Possible concentrations: ")
        df = pd.DataFrame(concentration.T, columns= element_list)
        display(df)
    return element_list, element_max, concentration   


def check_column_names(data,target,element_list):
    element_data = data.columns.drop(target)

    # Find the different column names
    columns_only_in_data = set(element_data).difference(element_list)

    # Find columns in df2 but not in df1
    columns_only_in_parameter = set(element_list).difference(element_data)

    try:
        if columns_only_in_data or columns_only_in_parameter:
            print("- Columns only in data files:", columns_only_in_data)
            print("- Columns only in parameter files:", columns_only_in_parameter)
            raise ValueError("Columns names are not matched, please modify parameter column names")
        else:
            print("All column names matched!")
    except ValueError as e:
        text = "{}: {}".format(type(e).__name__, e)
        print("\033[1;91m{}\033[0m".format(text))


#def normalize_data(data):
#    data_array = np.concatenate(data,axis = 0)
#    X = data_array[:,:-2]
#    y = data_array[:,-2]
#    scaler = MinMaxScaler()
#    scaled = scaler.fit_transform(X)
#    return scaled, y

def import_split_data(data_folder, element_list, target, type = 'first', idx = (102,204), seed = None):
    data = import_data(data_folder, verbose = False)
    if type == 'first':
        train = data.loc[idx[0]:idx[1]]
        data = data.drop(range(idx[0],idx[1]))

        X_train, y_train = train[element_list], train[target[0]]
        X_pool, y_pool = data[element_list], data[target[0]]

    if type == 'random':
        X_pool, y_pool = data[element_list], data[target[0]]
        X_pool, X_train, y_pool, y_train = train_test_split(X_pool, y_pool, test_size=0.1, random_state=seed)

    X_pool, X_test, y_pool, y_test = train_test_split(X_pool, y_pool, test_size=0.11, random_state=seed)

    return X_train, X_pool, X_test, y_train, y_pool, y_test


def normalized(X_train = [], X_test = [], X_pool = []):
    scaler = MaxAbsScaler()
    scaler.fit(X_train)
    X_normalized = scaler.transform(X_train)
    X_test_normalized, X_pool_normalized = [], []
    
    if len(X_test) > 0:
        X_test_normalized = scaler.transform(X_test)

    if len(X_pool) > 0:
        X_pool_normalized = scaler.transform(X_pool)

    return X_normalized, X_test_normalized, X_pool_normalized


def find_outlier(experiment,condi):
    if sum(condi) > 1:  
        result= []
    else: 
        result=experiment[~condi]
    return result


def remove_outlier_strict(sample):
    import numpy as np
    if not isinstance(sample, np.ndarray):
        sample = np.array(sample)

    mean = np.mean(sample, axis = 1)
    mean = np.tile(mean,(sample.shape[1], 1)).T

    std = np.std(sample, axis = 1)
    std = np.tile(std,(sample.shape[1], 1)).T

    z_scores = np.abs((sample-mean)/std)
#    outliers_z = np.abs(z_scores) > threshold
    result = []
    for i in range(len(z_scores)):
        z_row = z_scores[i]
        data_row = sample[i]
        std_row = std[i][0]
        if std_row > 0.2:
            condi = z_row > 1
        else:
            condi = z_row > 1.4
        result.append(find_outlier(data_row,condi))
    return result

def remove_outlier(sample, threshold = 0.2):
    import numpy as np
    if not isinstance(sample, np.ndarray):
        sample = np.array(sample)

    mean = np.mean(sample, axis = 1)
    mean = np.tile(mean,(sample.shape[1], 1)).T

    std = np.std(sample, axis = 1)
    std = np.tile(std,(sample.shape[1], 1)).T

    z_scores = np.abs((sample-mean)/std)
#    outliers_z = np.abs(z_scores) > threshold
    result = []
    for i in range(len(z_scores)):
        z_row = z_scores[i]
        data_row = sample[i]
        std_row = std[i][0]
        if std_row > threshold:
            condi = z_row > 1
            result.append(find_outlier(data_row,condi))
        else:
            result.append(data_row)
    return result


def flatten_X_y(X, y):
    X_flat = []
    for i in range(X.shape[0]):
        x_loop = X[i]
        y_loop = y[i]
        for _ in y_loop:
            X_flat.append(x_loop)
    X_flat = np.array(X_flat)
    y_flat = np.array([element for sublist in y for element in sublist])
    
    idx = np.isnan(y_flat)
    y_flat = y_flat[~idx]
    X_flat = X_flat[~idx]

    return X_flat, y_flat


def average_and_drop_na(X,y):
    if len(y)== 0:
        return X, y
    
    y = np.nanmean(y, axis = 1)
    idx = np.isnan(y)
    y = y[~idx]
    X = X[~idx]
    return X, y


def split_and_flatten(medium, yield_array, ratio = 0.2, seed = None, flatten = True):

    medium = np.array(medium)
    yield_array = np.array([np.pad(arr, (0, len(max(yield_array, key=len)) - len(arr)), 'constant', constant_values=np.nan) for arr in yield_array])
    _, unique_indices = np.unique(medium, axis=0, return_index=True)
    if len(unique_indices) != len(medium):
        print('BE CAREFUL! medium has repeatition')

    # Split 
    if ratio == 0:
        X_train, X_test = medium, np.array([])
        y_train, y_test = yield_array, np.array([])
    else:
        train_indices, test_indices = train_test_split(unique_indices, test_size=ratio, random_state=seed)
        X_train, X_test = medium[train_indices],  medium[test_indices]
        y_train = [yield_array[i] for i in train_indices]   # yield_array is list sometime
        y_test = [yield_array[i] for i in test_indices]

    # Flatten 
    if flatten:
       X_train, y_train = flatten_X_y(X_train, y_train) 
    else: 
        X_train, y_train = average_and_drop_na(X_train, y_train)
    
    X_test, y_test = flatten_X_y(X_test, y_test)
    return X_train, X_test, y_train, y_test

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2)**2, axis = 1))


#########################################################################################
def plot_loss(model_list, highlight=False):
    if not isinstance(model_list, list):
        model_list = [model_list]

    num_models = len(model_list)
    num_cols = round(np.sqrt(num_models))
    num_rows = (num_models + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 8))
    for i in range(num_cols*num_rows): # delete empty subplots
        if i >= num_models:
            fig.delaxes(axes.flatten()[i])
    
    min_loss_index = np.argmin([model.loss_ for model in model_list])

    for i, (model, ax) in enumerate(zip(model_list, np.ravel(axes))):
        losses = model.loss_curve_
        iterations = range(1, len(losses) + 1)

        if highlight:
            line_color = 'red' if i == min_loss_index else 'blue'
        else:
            line_color = 'blue'

        ax.plot(iterations, losses, color=line_color, label='Training loss')
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Loss')
        ax.set_title(f'Loss of Model {i + 1} : {round(model.loss_, 5)}')

    plt.tight_layout()
    plt.show()  


def plot_feature(data, label):
    data.hist(bins=4, color='blue', edgecolor='black', figsize=(8, 6))

    # Add labels and title
    plt.suptitle(f'Histograms for {label} features')
    plt.tight_layout()
    plt.show()


def plot_r2_curve(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    TRUE, PRED = y_true, y_pred
    sns.set(
        font="arial",
        palette="colorblind",
        style="whitegrid",
        font_scale=1.5,
        rc={"figure.figsize": (5, 5), "axes.grid": False},
    )
    sns.regplot(
        x=TRUE,
        y=PRED,
        fit_reg=0,
        marker="+",
        color="black",
        scatter_kws={"s": 40, "linewidths": 0.7},
    )
    plt.plot([min(TRUE), max(TRUE)], [min(TRUE), max(TRUE)], 
             linestyle='--', 
             color='blue',
             linewidth=1)
    plt.xlabel("Experiment ground truth ")
    plt.ylabel("Model prediction")
    plt.title(f'R2: {r2:.2f}', fontsize=14)
    plt.xlim(min(TRUE) - 0.2, max(TRUE) + 0.5)
    plt.ylim(min(PRED) - 0.2, max(PRED) + 0.5)
    plt.show()


def plot_hist_yield_std(mean, training_mean, std, training_std):
    # Plotting histograms side by side
    plt.figure(figsize=(10, 4))

    plt.subplot(2, 2, 1) 
    hist_range = [min(mean), max(mean)]
    plt.hist(mean, bins= 10, color='red', alpha=0.7, label='New points')
    plt.legend(prop={'size': 10})
    plt.title('Histogram of prediction yield')

    plt.subplot(2, 2, 3) 
    plt.hist(training_mean, bins=10, range=hist_range, color='black', alpha=0.5, label='Training points')
    plt.legend(prop={'size': 10})
    plt.title('Histogram of training yield')

    plt.subplot(2, 2, 2)  
    hist_range = [min(std), max(std)]
    plt.hist(std, bins=10, color='green', alpha=0.7, label='New points')
    plt.legend(prop={'size': 10})
    plt.title('Histogram of prediction std')

    plt.subplot(2, 2, 4)  
    plt.hist(training_std, bins=10, range=hist_range, color='black', alpha=0.5, label='Training points')
    plt.legend(prop={'size': 10})
    plt.title('Histogram of training std')
    plt.tight_layout()  # Adjust layout for better spacing
    plt.show()


def plot_hist_selected_yield(mean, selected_mean, title = 'Add title'):
    hist_range = [min(mean), max(mean)]
    plt.figure(figsize=(6, 3))
    plt.hist(mean, bins=20, color='green', alpha=0.7, label='Prediction')
    plt.hist(selected_mean, bins=20, range=hist_range, color='red', alpha=0.5, label='Selected points')
    
    plt.title(title)
    plt.ylabel('Frequency')
    plt.legend(loc='upper left')
    plt.tight_layout()  
    plt.show()


def plot_distance(ax, X_new_norm,ucb_top_norm, ucb, ratio_ucb, color, title):
    top_ucb = ucb_top_norm[0]
    distance_ucb = euclidean_distance(top_ucb, ucb_top_norm)
    distance_all_ucb = euclidean_distance(top_ucb,X_new_norm)

    ax.scatter(distance_all_ucb, ucb, color= 'grey', alpha = 0.5, label='Unselected points')
    ax.scatter(distance_ucb, ratio_ucb, color= color, alpha = 0.5, label='Selected points')
    ax.set_title(title)
    ax.legend(loc='upper right',  prop={'size': 10})


def plot_selected_point(ax, y_pred, std_pred, condition, title):
    # Specify the X positions where you want different colors
    position = np.where(np.isin(y_pred, condition))[0]
    selected_std = std_pred[position]
    selected_y = y_pred[position]

    # not selected points
    y_not = np.delete(y_pred,position)
    std_not = np.delete(std_pred,position)

    # Create a scatter plot with different colors at special X positions
    ax.scatter(y_not, std_not, c='grey', label='Unselected points', alpha = 0.5)
    ax.scatter(selected_y, selected_std, c='red', alpha = 0.5, label='Selected_point')

    # Customize the plot
    ax.set_title(title)
    ax.set_xlabel('Predicted yield')
    ax.set_ylabel('Predicted std')
    ax.legend(loc='upper left',  prop={'size': 10})


def plot_heatmap(ax, X_new_norm, y_pred, element_list, title):
    # Get the indices that would sort the array based on the values list
    n = len(y_pred)
    sorted_indices = np.argsort(y_pred)
    # Use the sorted indices to rearrange the rows of the array
    sorted_X = X_new_norm[sorted_indices]
    sorted_X = sorted_X.T

    # Create a heatmap
    heatmap=ax.imshow(sorted_X, aspect='auto', cmap='viridis')

    ax.set_xticks([])
    ax.set_title(f"{title}, sample = {n}", size = 12)
    ax.set_yticks(np.arange(len(element_list)), element_list, size = 12)
    cbar = plt.colorbar(heatmap, ax=ax)
    cbar.set_label('ratio with max concentrations', size = 12)

def plot_rmse(model):
    scores = model.cv_score
    score_list = scores[scores.rank_test_score == 1].iloc[:, 6:-3]
    score_list = np.array(score_list)*-1
    plt.hist(score_list[0], bins = 20, color='orange')
    plt.title(f'Histogram of RMSE during cross validation, mean = {round(np.mean(score_list),2)}', size = 12)


def plot_each_round(y,size_list, predict = False):
    # Calculate the cumulative sum of the size_list
    cumulative_sizes = np.cumsum(size_list)

    # Split the array into subarrays based on the cumulative sizes
    subarrays = np.split(y, cumulative_sizes)

    # Flatten each subarray
    flattened_arrays = [subarray.flatten() for subarray in subarrays]

    # Create a DataFrame 
    y_by_file = {}
    for i in range(len(size_list)):
        name = 'round ' + str(i)
        y_by_file[name] = flattened_arrays[i]

    y_by_file = pd.DataFrame.from_dict(y_by_file, orient='index').transpose()

    boxprops = dict(linewidth=0)
    medianprops = dict(linewidth=1, color='red', linestyle='dashed')
    ax = sns.boxplot(y_by_file, color = 'yellow', width=0.3, boxprops=boxprops, medianprops= medianprops)

    # Add markers for the maximum values
    #max_values = y_by_file.max()
    #for i, value in enumerate(max_values):
    #    ax.annotate(f'{value:.2f}', (i, value), textcoords="offset points", xytext=(0,5),
    #                ha='center', fontsize=8, color='black')

    # Add markers for the median values
    median_values = y_by_file.median()
    for i, value in enumerate(median_values):
        ax.annotate(f'{value:.2f}', (i, value), textcoords="offset points", xytext=(0,3),
                    ha='center', fontsize=8, color='black')

    if predict:
        # Get the last box patch
        last_box = ax.patches[-1]

        # Change the color of the last box
        last_box.set_facecolor('silver')

    plt.ylabel('Yield')
    plt.title('Yield evolution through each active learning query')
    # Show the plot
    plt.show()


def plot_train_test(train, test, element_list):
    test = pd.DataFrame(test, columns= element_list)
    train = pd.DataFrame(train, columns= element_list)

    # Plot histograms for each column in both DataFrames on the same figure
    no_element = len(element_list)
    nrows = int(np.sqrt(no_element))
    ncols = no_element//nrows
    _, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 6))

    for i, column in enumerate(element_list):
        row = i // ncols
        col = i % ncols
        hist_range = [min(train[column]), max(train[column])]
        axes[row, col].hist(train[column], alpha=0.3, label='Previous data', bins=10)
        axes[row, col].hist(test[column], range=hist_range, alpha=1, label='New data', bins=10)
        axes[row, col].set_title(column)
        axes[row, col].legend()
plt.tight_layout()
plt.show()