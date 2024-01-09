import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import r2_score


class MLPEnsemble:
    def __init__(self, 
                 hidden_layer_sizes=(100,),  
                 early_stopping = True,
                 learning_rate="adaptive", 
                 max_iter=20000,
                 random_state=None):
        
        self.models = []
        for hidden_size in hidden_layer_sizes:
            model = MLPRegressor(hidden_layer_sizes=hidden_size,
                                 learning_rate=learning_rate,
                                 max_iter=max_iter,
                                 early_stopping= early_stopping,
                                 random_state=random_state)
            self.models.append(model)

    def fit(self, X, y):
        # Normalize the input data during training 
        for model in self.models:
            model.fit(X, y)    
        return self

    def predict(self, X):
        predictions = []
        for model in self.models:
            predictions.append(model.predict(X))
        return predictions
    

def train_repeat(X,y,
                   hidden_layer_sizes,
                   nb_repeat = 2,
                   verbose = False):

    models = []
    for _ in range(nb_repeat):
        custom_mlp = MLPEnsemble(hidden_layer_sizes=hidden_layer_sizes, max_iter=20000, random_state=None)
        custom_mlp.fit(X,y)

        for i, hidden_size in enumerate(hidden_layer_sizes):
            models.append(custom_mlp.models[i])
            
            if verbose:
                print(f"Loss of {hidden_size}- architecture: {custom_mlp.models[i].loss_}") 

    return models


def pick_lowest_loss(model_list):

    losses = []
    for i in range(len(model_list)):
        losses.append(model_list[i].loss_) 

    try:
        best_index = losses.index(min(losses))
        best_model = model_list[best_index]
    except ValueError:
        print("Input is empty list without models")

    return best_model

###################################################################
def check_params(self, params):
    params = params if params else {}
    if all(not isinstance(value, list) for value in params.values()):
        self.params_search = False
        self.model_params = params
    else:
        self.params_search = True 
        self.model_params = {key: [value] if not isinstance(value, list) else value for key, value in params.items()}
    return self 


class EnsembleModels:
    def __init__(self, n_folds=5, model_type ='', params=None):
        check_params(self, params = params)
        self.n_folds = n_folds
        self.model_type = model_type
        self.models = []
        self.score = []    
            
    def create(self, params=None):
        if self.params_search:
            model_types = {
#                'linear': ElasticNet,
                'linear' : Ridge,
                'svm': SVR,
                'xgboost': XGBRegressor,
                'mlp': MLPRegressor,
                'gp' : GaussianProcessRegressor
            }
        else:
            model_types = {
                'linear': lambda: Ridge(**params),
#                'linear' : LinearRegression,
                'svm': lambda: SVR(**params),
                'xgboost': lambda: XGBRegressor(**params),
                'mlp': lambda: MLPRegressor(**params),
                'gp' : lambda: GaussianProcessRegressor(**params)
            }

        if self.model_type not in model_types:
            raise ValueError("Invalid model type")

        return model_types[self.model_type]()
    
    def train(self, X, y):
        # Convert X and y to NumPy arrays if they are not already
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)

        # Create KFold cross-validator
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        # Hyperparameter search
        if self.params_search:
            model = self.create()  # Create a model with default parameters
            grid_search = GridSearchCV(model, self.model_params, cv=kf, n_jobs=-1, scoring='r2')
            grid_search.fit(X, y)

            # Use the best parameters found by the search
            best_params = grid_search.best_params_
            print(f"Best hyperparameters: {best_params}")
            self.model_params = best_params
        else:
            print('No grid-search for hyperparameters')


        # Perform cross-validation and store models
        for i, (train_index, test_index) in enumerate(kf.split(X)):
            X_fold, X_validation = X[train_index], X[test_index]
            y_fold, y_validation = y[train_index], y[test_index]

            # Create a new model based on the specified type
            model = self.create(params=self.model_params)

            # Fit the model
            model.fit(X_fold, y_fold)

            # Store the model
            self.models.append(model)

            # Make predictions on the test set
            y_pred = model.predict(X_validation)
            self.score.append(r2_score(y_validation, y_pred))


    def predict(self, X):
        # Make predictions using each model
        predictions = []
        for model in self.models:
            predictions.append([model.predict(X)])

        # Calculate and store the standard deviation of predictions
        pred_std = np.std(predictions, axis=0)
        pred_mean = np.mean(predictions, axis=0)

        return pred_mean.ravel(), pred_std.ravel()
    

######################################################################
# gaussian proess class for bayersian opt

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.metrics import mean_squared_error

class GaussianProcessBayer:
    def __init__(self, params =None, cv_folds=5):
        """
        Initialize the Gaussian Process Regression model.

        Parameters:
        - kernels: List of kernel functions to be considered in grid search.
        - alpha: Small positive constant to ensure numerical stability.
        - n_restarts_optimizer: Number of restarts for hyperparameter optimization.
        - cv_folds: Number of cross-validation folds.
        """
        if params is None:
            params = {'kernel':[ 
                1.0 * RBF(length_scale=1.0),
                1.0 * C(1.0) * RBF(length_scale=1.0),
                1.0 * Matern(length_scale=1.0, nu=1.5),
                1.0 * Matern(length_scale=1.0, nu=2.5)
                ],
                'alpha': [1e-5]
                }

        self.params = params
        self.cv_folds = cv_folds
        self.best_kernel = None
        self.gp = None

    def train(self, X, y):
        """
        Find the best kernel using grid search and cross-validation.
        Then train on the best parameters

        Parameters:
        - X: Input features.
        - y: Target values.
        """
        gp = GaussianProcessRegressor()
        grid_search = GridSearchCV(gp, self.params, cv=self.cv_folds, n_jobs=-1, scoring='r2')
        grid_search.fit(X, y)

        self.best_params = grid_search.best_params_
        print(f"Best hyperparameter found: {self.best_params}")
        self.gp = GaussianProcessRegressor(**self.best_params)
        self.gp.fit(X, y)


    def predict(self, X):
        """
        Make predictions using the Gaussian Process Regression model.

        Parameters:
        - X: Input features for prediction.

        Returns:
        - mean: Predicted mean values.
        - std: Predicted standard deviation values.
        """
        mean, std = self.gp.predict(X, return_std=True)
        return mean, std
    
#########################################################################################################
def predict_rf(model, X):
    # Get predictions from all trees
    tree_predictions = np.array([tree.predict(X) for tree in model.estimators_])
    mean = np.mean(tree_predictions, axis=0)
    std = np.std(tree_predictions, axis=0)
    return mean, std

def predict_gp(model, X):
    mean, std = model.predict(X, return_std=True)
    return mean, std

def predict_ensemble(models, X):
    # Make predictions using each model
    predictions = []
    for model in models:
        predictions.append([model.predict(X)])

    # Calculate and store the standard deviation of predictions
    pred_std = np.std(predictions, axis=0)
    pred_mean = np.mean(predictions, axis=0)

    return pred_mean.ravel(), pred_std.ravel()

from sklearn.gaussian_process.kernels import RBF, Matern, DotProduct, ConstantKernel, WhiteKernel

class BayersianModels:
    def __init__(self, n_folds=5, model_type ='', params=None):
        self.n_folds = n_folds
        self.model_type = model_type
        self.models = []
        self.score = []
        if params == None:
#            raise ValueError("Empty parameters dictionary!")
            self.params = {'kernel': [
                RBF() + WhiteKernel(),
                Matern(length_scale=1, nu=1.5)+ WhiteKernel(),
                RBF() + ConstantKernel() + WhiteKernel(),
                Matern(length_scale=1, nu=1.5)+ ConstantKernel() + WhiteKernel(),
                Matern(length_scale=1, nu=1.5)+ RBF() + WhiteKernel(),
                Matern(length_scale=1, nu=1.5)+ DotProduct() + WhiteKernel(),
                RBF()+ DotProduct() + WhiteKernel(),
                RBF()*DotProduct() + WhiteKernel(),
                RBF()+ DotProduct() + ConstantKernel() + WhiteKernel(),
                RBF()*DotProduct() + ConstantKernel() + WhiteKernel()
                        ]}            
            self.model_type = 'gp'
        else:    
            self.params = params

    def create(self, params = None):
        model_types = {
            'rf' : lambda: RandomForestRegressor(**params),
            'xgboost': lambda: XGBRegressor(**params),
            'mlp': lambda: MLPRegressor(**params),
            'gp' : lambda: GaussianProcessRegressor(**params)
        }

        if self.model_type not in model_types:
            raise ValueError("Invalid model type")

        return model_types[self.model_type]()
    
    def train(self, X, y, verbose = True):
        # Convert X and y to NumPy arrays if they are not already
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)

        # Hyperparameter search
        model = self.create(self.params)  # Create a model with default parameters
        grid_search = GridSearchCV(model, self.params, cv = self.n_folds, n_jobs=-1, scoring='neg_root_mean_squared_error')

        grid_search.fit(X, y)
        self.best_params = grid_search.best_params_
        self.cv_score = pd.DataFrame(grid_search.cv_results_)

        if verbose:
            print(f"Best hyperparameter found: {self.best_params}")

        # Refit best scores
        if self.model_type in ['gp', 'rf']:
            #train on best hyperpara
            model = self.create(params=self.best_params)
            self.model = model.fit(X, y)

        if self.model_type in ['xgboost','mlp']:
            self.model = []
            for _ in range(20):
                model = self.create(params=self.best_params)
                self.model.append(model.fit(X, y))


    def predict(self,X):
        model_dict = {'rf': lambda: predict_rf(self.model,X),
                      'gp': lambda:predict_gp(self.model, X),
                      'xgboost': lambda: predict_ensemble(self.model, X),
                      'mlp': lambda: predict_ensemble(self.model,X)}

        return model_dict[self.model_type]()

    