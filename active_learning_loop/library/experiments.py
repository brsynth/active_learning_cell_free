import os
import numpy as np
import pandas as pd
import json
import skopt
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import MinMaxScaler
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF

from library.utils import *
from library.model import *
from library.blacbox import *
from library.active_learning import *

##### INPUT
#parameter_file = "data\\parameter_lipase_w4.csv"
first = False
seed = 13

parameter_file = "data\\parameter_lipase_modify_name.csv"
data_folder = "data\\no_controls"
target = ['yield', 'yield_std']

interation = 1
nb_ensemble = 8
nb_repeat = 4
hidden_layer_sizes = [(20, 100, 100, 20),
                    (20,100, 100,10),
                    (20,100, 100,20),
                    (10,100, 100,10),
                    (20,100, 100,50),
                    (100, 100,20),
]

nb_new_data = 100
visual = True

linear_params = {
    'alpha': [0, 0.1, 0.5, 1],
#    'l1_ratio' : [0, 0.25, 0.5, 0.75, 1]
}

mlp_params = {
    'hidden_layer_sizes' : [(32,64),(64),(32),(10)],
    'early_stopping' : [True],
    'learning_rate' : ["adaptive"], 
    'max_iter' : [2000],
}

svm_params = {
    'kernel': ['linear', 'rbf'],
    'C': [ 1, 10],  # Regularization parameter
    'epsilon': [0.001, 0.5],  # Epsilon parameter controlling the width of the epsilon-tube
}

xgb_params = {
    'objective': ['reg:squarederror'],
    'learning_rate': [0.01, 0.1],  # Step size shrinkage used to prevent overfitting
    'n_estimators': [100, 500, 700],       # Number of boosting rounds
    'max_depth': [3, 5, 7],               # Maximum depth of a tree
#    'subsample': 0.8,              # Subsample ratio of the training instances
#    'colsample_bytree': [0.8, 1.0],       # Subsample ratio of columns when constructing each tree
#    'gamma': [0, 0.1, 0.2],               # Minimum loss reduction required to make a further partition on a leaf node
}

gp_params = {
    'kernel': [DotProduct() + WhiteKernel(), 
               RBF + WhiteKernel(),
               RBF(length_scale=0.5, length_scale_bounds=(1e-1, 10.0)),
                1.0 * RBF(length_scale=1.0),
                1.0 * C(1.0) * RBF(length_scale=1.0),
                1.0 * Matern(length_scale=1.0, nu=1.5),
                1.0 * Matern(length_scale=1.0, nu=2.5)
                ]
                }

rf_params = {
    'n_estimators': [10, 50],
    'max_depth': [5, 10],
}

element_list, element_max, sampling_condition = import_parameter(parameter_file, verbose = True)
data = import_data(data_folder, verbose = False)
check_column_names(data,target,element_list)

# day_1 is the first train set
def preprocess(data_folder, element_list, target, type = 'first', idx = (102,204)):
    data = import_data(data_folder, verbose = False)
    if type == 'first':
        train = data.loc[idx[0]:idx[1]]
        data = data.drop(range(idx[0],idx[1]))

        X_train, y_train = train[element_list], train[target[0]]
        X_pool, y_pool = data[element_list], data[target[0]]

    if type == 'random':
        X_pool, y_pool = data[element_list], data[target[0]]
        X_pool, X_train, y_pool, y_train = train_test_split(X_pool, y_pool, test_size=0.1)

    X_pool, X_test, y_pool, y_test = train_test_split(X_pool, y_pool, test_size=0.11)    
    scaler = MaxAbsScaler()
    scaler.fit(X_train)
    X_normalized = scaler.transform(X_train)
    X_test_normalized = scaler.transform(X_test)
    return X_pool, X_train, X_test, X_normalized, y_pool, y_train, y_test, X_test_normalized, scaler



def repeat_test(data_folder, element_list, target, nb_repeat, nb_ensemble, model_type, params):
    
    models = EnsembleModels(n_folds=nb_ensemble, model_type= model_type, params= params)
    _, _, _, X_normalized, _, y_train, _, _, _= preprocess(data_folder, element_list, target, type = 'first')
    # Train the model
    models.train(X_normalized, y_train)
    r2 = []
    for _ in range(nb_repeat):
        _, _, _, _, _, _, y_test, X_test_normalized, _ = preprocess(data_folder, element_list, target, type = 'first')

        mean, _ = models.predict(X_test_normalized)
        r2.append(r2_score(y_test,mean))
                # Calculate mean and standard deviation
    mean_value = np.mean(r2)
    std_deviation = np.std(r2)

    return mean_value, std_deviation



def repeat_test_random(data_folder, element_list, target, nb_repeat, nb_ensemble, model_type, params):
    r2 = []
    for _ in range(nb_repeat):
        _, _, _, X_normalized, _, y_train, y_test, X_test_normalized, _ = preprocess(data_folder, element_list, target, type = 'random')
        models = EnsembleModels(n_folds=nb_ensemble, model_type= model_type, params= params)
         # Train the model
        models.train(X_normalized, y_train)
        
        mean, _ = models.predict(X_test_normalized)
        r2.append(r2_score(y_test,mean))
    
    # Calculate mean and standard deviation
    mean_value = np.mean(r2)
    std_deviation = np.std(r2)

    return mean_value, std_deviation

#test fixed train + random test
res = []
for model in dict:
    res.append(repeat_test(data_folder, element_list, target, 20, nb_ensemble,  model[0], model[1]))
    
res = np.array(res).T

#test random train + random test
res = []
for model in dict:
    res.append(repeat_test_random(data_folder, element_list, target, 20, nb_ensemble, model[0], model[1]))

res = np.array(res).T

name = ['linear + L2','mlp','svm','xgboost','gp']
colors = ['skyblue', 'lightgreen', 'salmon', 'khaki','pink']  # Different colors for each bar

# Create a bar plot with error bars and different colors
fig, ax = plt.subplots()
bars = ax.bar(name, res[0], yerr=res[1], capsize=5, color = colors)

# Add labels and title
ax.set_ylabel('R2 on random test set')
ax.set_title('Performance on random train/testing set')
ax.set_xticks(name)

# Add individual values on top of each bar
for bar, value in zip(bars, res[0]):
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, 0.6, round(value, 2), ha='center', va='top')

# Show the plot
plt.show()
####################################################################################################################################
def active_loop(data_folder, element_list, target, nb_ensemble, type, model, params, ratio, nb_new_data):
    X_train, X_pool, X_test, y_train, y_pool, y_test = import_split_data(data_folder, element_list, target, type = type)
    R2_scores = []
    yield_ = []
    for _ in range(100):
        if X_pool.shape[0] == 0:
            print("Pool is empty! Stop querying")
            break 

        X_normalized, X_test_normalized, X_pool_normalized = normalized(X_train, X_test, X_pool)

        models = EnsembleModels(n_folds=nb_ensemble, model_type= model, params= params)

        # evaluation
        models.train(X_normalized, y_train)
        mean, std = models.predict(X_test_normalized)
        score = r2_score(y_test,mean)
        R2_scores.append(score)

        # predict pool
        mean, std = models.predict(X_pool_normalized)
        ucb = mean + ratio*std

        #find best ucb and update pool
        ucb_top, y_top, X_pool, y_pool = find_update_top_element(X_pool, y_pool, ucb, nb_new_data, verbose = False)
        yield_.append(y_top.tolist())

        #update new X
        X_train = pd.concat([X_train, pd.DataFrame(ucb_top, columns= element_list)])
        y_train = pd.concat([pd.Series(y_train),pd.Series(y_top)])

    return R2_scores, yield_

# repeat through interation
nb_repeat = 3
dict = [['linear',linear_params],['mlp',mlp_params],['svm',svm_params],['xgboost',xgb_params]]
interation_score = {}
interation_yield = {}
for model in dict:
    repeat_score = []
    repeat_yield = []

    for i in range(nb_repeat):
        print(i + 1)
        scores, y_yield = active_loop(data_folder, element_list, target, 
                                nb_ensemble, type = 'first', 
                                model = model[0], params = model[1], 
                                ratio = 0, nb_new_data = 20)
        repeat_score.append(scores)
        repeat_yield.append(y_yield)
        
    interation_score[model[0]] = repeat_score
    interation_yield[model[0]] = repeat_yield


#converted_dict = {key: [[[item.tolist() for item in sublist] for sublist in sublist_list] for sublist_list in value] for key, value in interatio_yield.items()}

# Specify the file path
name = 'test_random_sample_20_ratio_0'
score_path = 'active_result/interation_'+ name +'_score.json'
yield_path = 'active_result/interation_'+ name +'_yield.json'

json.dump(interation_score, open(score_path, 'w' ) )
json.dump(interation_yield, open(yield_path, 'w' ) )

################################################################################################################################################################""""

def active_loop_query_bayer(n_sample, element_list,  model, model_params, nb_interation, nb_ensemble, ratio, nb_new_data, try_ratio):    
    # generate data
#    np.random.seed(42)
    X_train = pd.DataFrame(np.random.rand(n_sample, 11), columns=element_list)
    y_train = objective(X_train)
    
#    np.random.seed(54)
    X_test = pd.DataFrame(np.random.rand(n_sample, 11), columns=element_list)
    y_test = objective(X_test)

    R2_scores = []
    yield_ = []
    for _ in range(nb_interation):

        # modelling
        models = EnsembleModels(n_folds=nb_ensemble, model_type= model, params= model_params) # linear, mlp, svm, xgboost

        # evaluation
        models.train(X_train, y_train)

        # models test
        mean, _ = models.predict(X_test)
        score = r2_score(y_test,mean)
        R2_scores.append(score)

        # sampling new data
        X_new= sampling_without_repeat(sampling_condition, nb_sample = nb_new_data*try_ratio, exited_data=X_train)

        # predict on new data
        mean, std = models.predict(X_new)

        # calculated ucb and find top 
        ucb = mean + ratio*std
        ucb_top, _, _ = find_top_element(X_new, mean ,ucb, nb_new_data, return_ratio= True, verbose = False)

        #query by objective function
        scaler = MinMaxScaler()
        ucb_normalize = scaler.fit_transform(ucb_top)
        y_true = objective(ucb_normalize)
        print(f'Highest yield found: {max(y_true)}')
        yield_.append(y_true.tolist())

        # update X_train/ y_train
        X_train = pd.concat([X_train, pd.DataFrame(ucb_top, columns=element_list)], axis=0)
        y_train = pd.concat([pd.Series(y_train),pd.Series(y_true)])
    
    return R2_scores, yield_

# repeat through interation
nb_repeat = 5
dict = [['linear',linear_params],['mlp',mlp_params],['svm',svm_params],['xgboost',xgb_params], ['gp', gp_params]]

interation_score = {}
interation_yield = {}
for mod in dict:
    repeat_score = []
    repeat_yield = []

    for i in range(nb_repeat):
        print(i + 1)
        scores, y_yield = active_loop_query_bayer(100, element_list,  
                                                 model = mod[0], model_params = mod[1],
                                                  nb_interation = 10, nb_ensemble = 5, 
                                                  ratio = 1.14, nb_new_data = 100, try_ratio = 100)
        repeat_score.append(scores)
        repeat_yield.append(y_yield)
        
    interation_score[mod[0]] = repeat_score
    interation_yield[mod[0]] = repeat_yield


# Specify the file path
name = 'bayer_objectiv_ubc_100'
score_path = 'active_result/interation_'+ name +'_score.json'
yield_path = 'active_result/interation_'+ name +'_yield.json'

json.dump(interation_score, open(score_path, 'w' ) )
json.dump(interation_yield, open(yield_path, 'w' ) )



n_sample = 100
try_ratio = 100
nb_interation = 10
nb_new_data = 10

def active_gaussian(n_sample, element_list, gp_params, acquisition , nb_interation, nb_new_data, try_ratio):
    # generate data
#    np.random.seed(42)
    X_train = pd.DataFrame(np.random.rand(n_sample, 11), columns=element_list)
    y_train = objective(X_train)

#    np.random.seed(54)
    X_test = pd.DataFrame(np.random.rand(n_sample, 11), columns=element_list)
    y_test = objective(X_test)

    R2_scores = []
    yield_ = []
    for _ in range(nb_interation):
        # Instantiate the GaussianProcessModel
        model = GaussianProcessBayer(params = gp_params, cv_folds=5)

        # Train the model using the best kernel
        model.train(X_train, y_train)

        # Make predictions on test data
        mean, std = model.predict(X_test)

        # Example evaluation using mean squared error
        r2 = r2_score(y_test, mean)
        R2_scores.append(r2)
        print(f"Mean Squared Error on Test Data: {r2}")

        # sampling new data
        X_new= sampling_without_repeat(sampling_condition, nb_sample = nb_new_data*try_ratio, exited_data=X_train)

        # predict on new data
        mean, std = model.predict(X_new)

        # calculated informativeness using acquisation function and find top
        y_best = max(y_train)
        if acquisition == 'pi':
            acq = probability_of_improvement(mean, std, y_best)
        if acquisition == 'ei':
            acq = expected_improvement(mean, std)
        if acquisition == 'ucb':
            acq = mean + 1.14*std

        acq_top, _, _ = find_top_element(X_new, mean ,acq, nb_new_data, return_ratio= True, verbose = False)

        #query by objective function
        scaler = MinMaxScaler()
        ucb_normalize = scaler.fit_transform(acq_top)
        y_true = objective(ucb_normalize)
        print(f'Highest yield found: {max(y_true)}')
        yield_.append(y_true.tolist())

        # update X_train/ y_train
        X_train = pd.concat([X_train, pd.DataFrame(acq_top, columns=element_list)], axis=0)
        y_train = pd.concat([pd.Series(y_train),pd.Series(y_true)])
    
    return R2_scores, yield_

# repeat through interation
nb_repeat = 20
dict = [['gp', gp_params]]
acquisition = 'ucb'

interation_score = {}
interation_yield = {}
for mod in dict:
    repeat_score = []
    repeat_yield = []

    for i in range(nb_repeat):
        print(i + 1)
        scores, y_yield = active_gaussian(n_sample, 
                                          element_list, 
                                          gp_params, 
                                          acquisition =acquisition, 
                                          nb_interation = 100, 
                                          nb_new_data = 1, 
                                          try_ratio= 100)
        repeat_score.append(scores)
        repeat_yield.append(y_yield)
        
    interation_score[mod[0]] = repeat_score
    interation_yield[mod[0]] = repeat_yield


# Specify the file path
name = 'bayer_gaussian_10_' + acquisition
score_path = 'active_result/interation_'+ name +'_score.json'
yield_path = 'active_result/interation_'+ name +'_yield.json'

json.dump(interation_score, open(score_path, 'w' ) )
json.dump(interation_yield, open(yield_path, 'w' ) )

##########################################################################################################
# Testing gold regressors
# Split X/y
X = data[element_list]
y = data[target[0]]

from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=53)

# normalize data
X_normalized, X_test_normalized, _ = normalized(X_train, X_test)

best_dict = {'learning_rate': 0.01, 'max_depth': 7, 'n_estimators': 500, 'objective': 'reg:squarederror'}
#best_dict = {'early_stopping': True, 'hidden_layer_sizes': (32, 64), 'learning_rate': 'adaptive', 'max_iter': 1000}
#best_dict = {'C': 10, 'epsilon': 0.1, 'kernel': 'rbf'}

gold = XGBRegressor(**best_dict)
#gold = MLPRegressor(**best_dict)
#gold = SVR(**best_dict)

gold.fit(X_normalized, y_train)

# Make predictions on the test set
# tree_predictions = gold.apply(X_test_normalized)
#y_pred = np.mean(tree_predictions, axis=1)
#std = np.std(tree_predictions, axis=1)
y_pred = gold.predict(X_test_normalized)
score = r2_score(y_test, y_pred)
print("Model performances during testing")
print(score)


ratio = 1.14
try_ratio = 100 
nb_interation = 10

def active_loop_query_gold(data_folder, element_list, target,  model, model_params, gold_model, nb_interation, nb_ensemble, ratio, nb_new_data, try_ratio):    
    X_train, _, X_test, y_train, _, y_test = import_split_data(data_folder, element_list, target , type = 'first')
    R2_scores = []
    yield_ = []
    for _ in range(nb_interation):
        # normalize data
        X_normalized, X_test_normalized, _ = normalized(X_train, X_test)

        # modelling
        models = EnsembleModels(n_folds=nb_ensemble, model_type= model, params= model_params) # linear, mlp, svm, xgboost

        # evaluation
        models.train(X_normalized, y_train)

        # models test
        mean, _ = models.predict(X_test_normalized)
        score = r2_score(y_test,mean)
        R2_scores.append(score)

        # sampling new data
        X_new= sampling_without_repeat(sampling_condition, nb_sample = nb_new_data*try_ratio, exited_data=X_train)

        # predict on new data
        _, X_new_normalized, _ = normalized(X_train, X_new)
        mean, std = models.predict(X_new_normalized)

        # calculated ucb and find top 
        ucb = mean + ratio*std
        ucb_top, _, ratio_list = find_top_element(X_new, mean ,ucb, nb_new_data, return_ratio= True, verbose = True)

        #query by gold regressor
        _, ucb_top_normalized, _ = normalized(X_train, ucb_top)
        y_gold = gold_model.predict(ucb_top_normalized)
        yield_.append(y_gold.tolist())

        # update X_train/ y_train
        X_train = pd.concat([X_train, pd.DataFrame(ucb_top, columns=element_list)], axis=0)
        y_train = pd.concat([pd.Series(y_train),pd.Series(y_gold)])
    
    return R2_scores, yield_


# repeat through interation
nb_repeat = 20
#dict = [['linear',linear_params],['mlp',mlp_params],['svm',svm_params],['xgboost',xgb_params], ['gp', gp_params]]
dict = [['mlp',mlp_params]]
interation_score = {}
interation_yield = {}
for model in dict:
    repeat_score = []
    repeat_yield = []

    for i in range(nb_repeat):
        print(i + 1)
        scores, y_yield = active_loop_query_gold(data_folder, element_list, target,  
                                                 model = model[0], model_params = model[1], gold_model = gold,
                                                  nb_interation = nb_interation, nb_ensemble = nb_ensemble, 
                                                  ratio = 1.14, nb_new_data = 100, try_ratio = 100)
        repeat_score.append(scores)
        repeat_yield.append(y_yield)
        
    interation_score[model[0]] = repeat_score
    interation_yield[model[0]] = repeat_yield

# Specify the file path
name = 'gold_sample_100_only_mlp'
score_path = 'active_result/interation_'+ name +'_score.json'
yield_path = 'active_result/interation_'+ name +'_yield.json'

json.dump(interation_score, open(score_path, 'w' ) )
json.dump(interation_yield, open(yield_path, 'w' ) )



# repeat through interation
nb_repeat = 5
dict = [['linear',linear_params],['mlp',mlp_params],['svm',svm_params],['xgboost',xgb_params], ['gp', gp_params]]
interation_score = {}
interation_yield = {}
for model in dict:
    repeat_score = []
    repeat_yield = []

    for i in range(nb_repeat):
        print(i + 1)
        scores, y_yield = active_loop_query_gold(data_folder, element_list, target,  
                                                 model = model[0], model_params = model[1], gold_model = gold,
                                                  nb_interation = 50, nb_ensemble = nb_ensemble, 
                                                  ratio = 1.14, nb_new_data = 10, try_ratio = 100)
        repeat_score.append(scores)
        repeat_yield.append(y_yield)
        
    interation_score[model[0]] = repeat_score
    interation_yield[model[0]] = repeat_yield


# Specify the file path
name = 'gold_sample_10_ratio_1.14'
score_path = 'active_result/interation_'+ name +'_score.json'
yield_path = 'active_result/interation_'+ name +'_yield.json'

json.dump(interation_score, open(score_path, 'w' ) )
json.dump(interation_yield, open(yield_path, 'w' ) )



def active_test_mathilde(data_folder, element_list, target,  hidden_layer_sizes, gold_model, nb_interation, nb_ensemble, ratio, nb_new_data, try_ratio):
    X_train, _, X_test, y_train, _, y_test = import_split_data(data_folder, element_list, target , type = 'first')
    R2_scores = []
    yield_ = []
    for _ in range(nb_interation):
        # normalize data
        X_normalized, X_test_normalized, _ = normalized(X_train, X_test)

        # train emsemble
        ensemble_models = []
        for i in range(nb_ensemble):
            repeat_models = train_repeat(X_normalized, y_train,
                    hidden_layer_sizes,
                    nb_repeat = nb_repeat,
                    verbose = False)
    
            best_model = pick_lowest_loss(repeat_models)
            ensemble_models.append(best_model)

        # models test
        y_pred = []
        for i in range(len(ensemble_models)):
            model = ensemble_models[i]
            y_pred.append(model.predict(X_test_normalized))

        y_pred = np.array(y_pred)
        mean = np.average(y_pred, axis=0)
        score = r2_score(y_test,mean)
        R2_scores.append(score)

        # sampling new data
        X_new= sampling_without_repeat(sampling_condition, nb_sample = nb_new_data*try_ratio, exited_data=X_train)

        # predict on new data
        y_pred = []
        _, X_new_normalized, _ = normalized(X_train, X_new)
        for i in range(len(ensemble_models)):
            model = ensemble_models[i]
            y_pred.append(model.predict(X_new_normalized))

        y_pred = np.array(y_pred)
        mean = np.average(y_pred, axis=0)
        std = np.std(y_pred, axis=0)

        # calculated ucb and find top 
        ucb = mean + ratio*std
        ucb_top, _, ratio_list = find_top_element(X_new, mean ,ucb, nb_new_data, return_ratio= True, verbose = True)

        #query by gold regressor
        _, ucb_top_normalized, _ = normalized(X_train, ucb_top)
        y_gold = gold_model.predict(ucb_top_normalized)
        yield_.append(y_gold.tolist())

        # update X_train/ y_train
        X_train = pd.concat([X_train, pd.DataFrame(ucb_top, columns=element_list)], axis=0)
        y_train = pd.concat([pd.Series(y_train),pd.Series(y_gold)])

    return R2_scores, yield_


# repeat through interation
nb_repeat = 3
#dict = [['linear',linear_params],['mlp',mlp_params],['svm',svm_params],['xgboost',xgb_params], ['gp', gp_params]]
dict = [['mlp']]

interation_score = {}
interation_yield = {}
for model in dict:
    repeat_score = []
    repeat_yield = []

    for i in range(nb_repeat):
        print(i + 1)
        scores, y_yield = active_test_mathilde(data_folder, element_list, target,  
                                                 hidden_layer_sizes, gold_model = gold,
                                                  nb_interation = 10, nb_ensemble = nb_ensemble, 
                                                  ratio = 1.14, nb_new_data = 100, try_ratio = 100)
        repeat_score.append(scores)
        repeat_yield.append(y_yield)
        
    interation_score[model[0]] = repeat_score
    interation_yield[model[0]] = repeat_yield


# Specify the file path
name = 'gold_sample_100_amir'
score_path = 'active_result/interation_'+ name +'_score.json'
yield_path = 'active_result/interation_'+ name +'_yield.json'

json.dump(interation_score, open(score_path, 'w' ) )
json.dump(interation_yield, open(yield_path, 'w' ) )


##########################################################################################################
# testing flatten vs none, outliers vs non

data_folder = "data\\lea_result"
target = ['yield_1', 'yield_2','yield_3', 'yield', 'yield_std']
name_list = ['yield_1', 'yield_2','yield_3']
parameter_file = "data\\lea\\params.csv"


# remove outliers
yield_array = np.array(data[name_list])
mean_0 = np.mean(yield_array, axis = 1)
std_0 = np.std(yield_array, axis = 1)

data_no = remove_outlier(yield_array, threshold = 0.2)
std = []
mean = []
for i in data_no:
    std.append(np.std(i))
    mean.append(np.mean(i))

std = np.array(std)

# Plot histograms side by side
plt.figure(figsize=(10, 5))
plt.hist(std_0, bins=20, color='green', alpha=0.5, label='before remove outliers')
plt.hist(std, bins=10, color='red', alpha=0.6, label='after remove outliers')

plt.title('Std distribution')
plt.xlabel('Std of repeatations')
plt.ylabel('Frequency')
plt.legend()

#remove ratio
n_full = len(yield_array.ravel())
n_rest = np.count_nonzero(~np.isnan(np.array([np.pad(arr, (0, len(max(data_no, key=len)) - len(arr)), 'constant', constant_values=np.nan) for arr in data_no]).ravel()))
n_rest*100/n_full

# Plot yields
plt.figure(figsize=(10, 5))
plt.hist(mean_0, bins=10, color='green', alpha=0.5, label='before remove outliers')
plt.hist(mean, bins=15, color='red', alpha=0.6, label='after remove outliers')

plt.title('Yield distribution')
plt.xlabel('Yield of repeatations')
plt.ylabel('Frequency')
plt.legend()

column_name = data.columns
element_list = [x for x in column_name if x not in target]
no_element = len(element_list)
medium = data.iloc[:,0:no_element]
X_train, X_test, y_train, y_test = split_and_flatten(medium, data_no, ratio = 0.2, flatten = False)

scaler = MaxAbsScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.fit_transform(X_test)

# histogram of X
train = pd.DataFrame(X_train, columns= element_list)
test = pd.DataFrame(X_test, columns=element_list)

# Plot histograms for each column in both DataFrames on the same figure
nrows = int(np.sqrt(no_element))
ncols = no_element//nrows
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 6))

for i, column in enumerate(element_list):
    row = i // ncols
    col = i % ncols
    axes[row, col].hist(train[column], alpha=0.3, label='Train', bins=10)
    axes[row, col].hist(test[column], alpha=1, label='Test', bins=10)
    axes[row, col].set_title(column)
    axes[row, col].legend()

plt.tight_layout()
plt.show()


gp_param = {
    'kernel': [
        DotProduct() + WhiteKernel(), 
        RBF() + WhiteKernel(),
#               RBF(),
#                1.0 * RBF(length_scale=1.0),
        1.0 * C(1.0) * RBF(length_scale=0) + WhiteKernel(),
        1.0 * Matern(length_scale=1, nu=1.5),
                ]
                }

mlp_param = {
    'hidden_layer_sizes' : [(32,64),(64),(32),(10),(40,40)],
    'early_stopping' : [True],
    'learning_rate' : ["adaptive"], 
    'max_iter' : [2000],
}

xgb_param = {
    'objective': ['reg:squarederror'],
    'learning_rate': [0.01, 0.1, 0.5],  # Step size shrinkage used to prevent overfitting
    'n_estimators': [10, 50, 100, 500, 1000],       # Number of boosting rounds
    'max_depth': [3, 5, 7, 10, 20,40],               # Maximum depth of a tree
}

rf_param = {
    'n_estimators': [10, 50, 100],
    'max_depth': [7, 10, 20],
}

param_dict = {
    'gp' : gp_param,
    'mlp': mlp_param,
    'xgboost' : xgb_param,
    'rf' : rf_param
}

model_list = ["gp",'mlp','xgboost','rf']
no_rep = 50 
result = pd.DataFrame(columns = model_list)
for i in range(no_rep):
    print(i+1)
    X_train, X_test, y_train, y_test = split_and_flatten(medium, data_no, ratio = 0.2, flatten = True)

    scaler = MaxAbsScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)
    res = []
    for name in model_list:
        model = BayersianModels(model_type = name, params= param_dict[name])
        model.train(X_train_norm, y_train)
        y_pred, std_pred = model.predict(X_test_norm)
        res.append(r2_score(y_test, y_pred))
    
    result = pd.concat([result, pd.DataFrame([res], columns=model_list)], axis=0)



file_name = 'Model_zscore_flat'
result.to_excel('mahnaz\\'+file_name+'.xlsx', index=False)
sns.boxplot(data = result)
# Show the plot
plt.title(file_name)
#plt.ylim(top = 1, bottom = -0.2)
plt.ylabel('r2')
plt.show()


# modelling 
gp_param = {
    'kernel': [
#        DotProduct() + WhiteKernel(), 
#        RBF() + WhiteKernel(),
#               RBF(),
#                1.0 * RBF(length_scale=1.0),
#        1.0 * C(1.0) * RBF(length_scale=0) + WhiteKernel(),
        1.0 * Matern(length_scale=1, nu=1.5) + WhiteKernel(noise_level=0.2),
#        1.0 * Matern(length_scale=1, nu=2.5)
                ]
                }

param_dict = {
    'gp' : gp_param,
    'mlp': mlp_param,
    'xgboost' : xgb_param,
    'rf' : rf_param
}
pd.set_option('display.max_rows', None) 
column = ['Marten', 'DotProduct', 'RBF', 'C']
repeat = 30
result = pd.DataFrame(columns = column)
for i in range(repeat):
    res = []
    X_train, X_test, y_train, y_test = split_and_flatten(medium, yield_array, ratio = 0.2, flatten = True)

    scaler = MaxAbsScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)

    model = BayersianModels(model_type = 'gp', params= {'kernel': [1.0 * Matern(length_scale=1, nu=1.5) + WhiteKernel()]})
    model.train(X_train_norm, y_train)
    y_pred, std_pred = model.predict(X_test_norm)
    res.append(r2_score(y_test, y_pred))

    model = BayersianModels(model_type = 'gp', params= {'kernel': [DotProduct() + WhiteKernel()]})
    model.train(X_train_norm, y_train)
    y_pred, std_pred = model.predict(X_test_norm)
    res.append(r2_score(y_test, y_pred))

    model = BayersianModels(model_type = 'gp', params= {'kernel': [RBF() + WhiteKernel()]})
    model.train(X_train_norm, y_train)
    y_pred, std_pred = model.predict(X_test_norm)
    res.append(r2_score(y_test, y_pred))

    model = BayersianModels(model_type = 'gp', params= {'kernel': [1.0 * C(1.0) * RBF(length_scale=0) + WhiteKernel()]})
    model.train(X_train_norm, y_train)
    y_pred, std_pred = model.predict(X_test_norm)
    res.append(r2_score(y_test, y_pred))

    result = pd.concat([result, pd.DataFrame([res], columns=column)], axis=0)


import scipy.stats as stats
stats.ttest_rel(result['flat'], result['average']) 
sns.boxplot(data = result)
plt.title('Compare input style for gaussian process')
plt.ylabel('r2')
plt.show()

import scipy.stats as stats
fvalue, pvalue = stats.f_oneway(result['Marten'], result['RBF'], result['DotProduct'])
print(fvalue, pvalue)

plt.figure(figsize = [10,5])
sns.boxplot(data = result)
plt.title('Compare kernels for gaussian process')
plt.ylabel('r2')
plt.show()


no_element = len(element_list)
medium = data.iloc[:,0:no_element]
X_train, X_test, y_train, y_test = split_and_flatten(medium, data_no, ratio = 0, flatten = False)

scaler = MaxAbsScaler()
X_train_norm = scaler.fit_transform(X_train)


model = BayersianModels(model_type = 'xgboost', params= xgb_param)
model.train(X_train_norm, y_train)
y_pred, std_pred = model.predict(X_train_norm)


plot_r2_curve(y_train, y_pred)

plt.hist(std_pred)

############################################################################
# experiment to find best model- best kernel for gp etc

model_list = ["gp",'mlp','xgboost','rf']
no_rep = 50 
result = pd.DataFrame(columns = model_list)
for i in range(no_rep):
    print(i+1)
    X_train, X_test, y_train, y_test = split_and_flatten(medium, yield_array, ratio = 0.2, flatten = True)

    scaler = MaxAbsScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)
    res = []
    for name in model_list:
        model = BayersianModels(model_type = name, params= param_dict[name])
        model.train(X_train_norm, y_train)
        y_pred, std_pred = model.predict(X_test_norm)
        res.append(r2_score(y_test, y_pred))
    
    result = pd.concat([result, pd.DataFrame([res], columns=model_list)], axis=0)


file_name = 'Model_coef_flat'
result = pd.read_excel('data\\lea_result\\'+file_name+'.xlsx')
ax = sns.boxplot(data = result)

# Calculate the average for each day
averages = result.mean()

# Add average annotations to each box
for i, col in enumerate(result.columns):
    ax.text(i, averages[col], f'Avg: {averages[col]:.2f}', ha='center', va='top', fontsize=9, color='black')

# Show the plot
plt.title(file_name)
plt.ylim(top = 1, bottom = -0.2)
plt.ylabel('r2')
plt.show()



# Calculate the average for each day
averages = result.mean()

# Add average annotations to each box
for i, col in enumerate(result.columns):
    ax.text(i, averages[col], f'Avg: {averages[col]:.2f}', ha='center', va='bottom', fontsize=9, color='black')

# Show the plot
plt.show()

# modelling 
gp_param = {
    'kernel': [
#        DotProduct() + WhiteKernel(), 
#        RBF() + WhiteKernel(),
#               RBF(),
#                1.0 * RBF(length_scale=1.0),
#        1.0 * C(1.0) * RBF(length_scale=0) + WhiteKernel(),
        1.0 * Matern(length_scale=1, nu=1.5) + WhiteKernel(noise_level=0.2),
#        1.0 * Matern(length_scale=1, nu=2.5)
                ]
                }

param_dict = {
    'gp' : gp_param,
    'mlp': mlp_param,
    'xgboost' : xgb_param,
    'rf' : rf_param
}
pd.set_option('display.max_rows', None) 
column = ['Marten', 'DotProduct', 'RBF', 'C']
repeat = 30
result = pd.DataFrame(columns = column)
for i in range(repeat):
    res = []
    X_train, X_test, y_train, y_test = split_and_flatten(medium, yield_array, ratio = 0.2, flatten = True)

    scaler = MaxAbsScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)

    model = BayersianModels(model_type = 'gp', params= {'kernel': [1.0 * Matern(length_scale=1, nu=1.5) + WhiteKernel()]})
    model.train(X_train_norm, y_train)
    y_pred, std_pred = model.predict(X_test_norm)
    res.append(r2_score(y_test, y_pred))

    model = BayersianModels(model_type = 'gp', params= {'kernel': [DotProduct() + WhiteKernel()]})
    model.train(X_train_norm, y_train)
    y_pred, std_pred = model.predict(X_test_norm)
    res.append(r2_score(y_test, y_pred))

    model = BayersianModels(model_type = 'gp', params= {'kernel': [RBF() + WhiteKernel()]})
    model.train(X_train_norm, y_train)
    y_pred, std_pred = model.predict(X_test_norm)
    res.append(r2_score(y_test, y_pred))

    model = BayersianModels(model_type = 'gp', params= {'kernel': [1.0 * C(1.0) * RBF(length_scale=0) + WhiteKernel()]})
    model.train(X_train_norm, y_train)
    y_pred, std_pred = model.predict(X_test_norm)
    res.append(r2_score(y_test, y_pred))

    result = pd.concat([result, pd.DataFrame([res], columns=column)], axis=0)


import scipy.stats as stats
stats.ttest_rel(result['flat'], result['average']) 
sns.boxplot(data = result)
plt.title('Compare input style for gaussian process')
plt.ylabel('r2')
plt.show()

import scipy.stats as stats
fvalue, pvalue = stats.f_oneway(result['Marten'], result['RBF'], result['DotProduct'])
print(fvalue, pvalue)

plt.figure(figsize = [10,5])
sns.boxplot(data = result)
plt.title('Compare kernels for gaussian process')
plt.ylabel('r2')
plt.show()

yield_array = np.array(data[name_list])
no_element = len(element_list)
medium = data.iloc[:,0:no_element]

