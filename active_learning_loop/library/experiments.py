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