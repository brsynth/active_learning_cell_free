import numpy as np
import pandas as pd
from scipy.stats import norm

def sample_new_combination(parameter, nb_sample = 100000, seed=None):
    samples = []
    np.random.seed(seed)
    for i in range(len(parameter)):
        choice = np.random.choice(parameter[i] , size = nb_sample, replace = True)
        samples.append(choice)
    samples = np.array(samples).transpose()
    return samples 


def sampling_without_repeat(sampling_condition, nb_sample, exited_data, seed = None):
     if not isinstance(exited_data, np.ndarray):
        exited_data = np.array(exited_data)

     new_comb = sample_new_combination(sampling_condition, nb_sample, seed=seed)

     #check if new data already exits and resample
     while True:
        matches = np.all(new_comb[:, np.newaxis] == exited_data, axis=-1)
        rows_to_drop = np.any(matches, axis=1)

        if not np.any(rows_to_drop):
                # No rows to be dropped, so we are done
                break
        
        # resampling the same repeated number
        nb_repeat = sum(rows_to_drop)
        resample = sample_new_combination(sampling_condition, nb_sample = nb_repeat, seed= None)
        new_comb = new_comb[~rows_to_drop]
        new_comb = np.concatenate([new_comb, resample])

     return new_comb


def find_top_element(X, y, condition, n, return_ratio = False, verbose = True):
        # Convert X and y to NumPy arrays if they are not already
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    if not isinstance(y, np.ndarray):
        y = np.array(y)

    # Sort the list in ascending order (smallest to largest)
    idx_top_n = np.argsort(-condition)[:n]
    choosen_X = X[idx_top_n,:]
    choosen_y = y[idx_top_n]

    if return_ratio:
        ratio = condition[idx_top_n]
#        ratio = choosen_y/(ucb - choosen_y)
    else:
        ratio = []

    if verbose:
        print(f"Maximum yield prediction = {max(choosen_y)}")
    return choosen_X, choosen_y, ratio


def find_update_top_element(X, y, condition, n, verbose = True):
        # Convert X and y to NumPy arrays if they are not already
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    if not isinstance(y, np.ndarray):
        y = np.array(y)

    # Sort the list in ascending order (smallest to largest)
    idx_top_n = np.argsort(-condition)[:n]
    choosen_X = X[idx_top_n,:]
    choosen_y = y[idx_top_n]
    
    unselected_X = np.delete(X, idx_top_n, axis=0)
    unselected_y = np.delete(y, idx_top_n)

    if verbose:
        print(f"Maximising sample has yield = {max(choosen_y)}")
        
    return choosen_X, choosen_y, unselected_X, unselected_y



def active_found(ensemble_models, sampling_condition, nb_new_data, X, scaler,verbose = True):
    X_new= sampling_without_repeat(sampling_condition, nb_sample = nb_new_data*5, exited_data=X)

    y_pred = []
    X_new_normalized = scaler.transform(X_new)
    for i in range(len(ensemble_models)):
        model = ensemble_models[i]
        y_pred.append(model.predict(X_new_normalized))

    y_pred = np.array(y_pred)
    y_mean = np.average(y_pred, axis=0)
    y_stdv = np.std(y_pred, axis=0)

    ucb = 1*y_mean + 1.14*y_stdv 

    print("For UCB:")
    ucb_top, _, ratio = find_top_element(X_new, y_mean,ucb, nb_new_data, return_ratio= True, verbose = verbose)
    print("For exploitation:")
    exploit_top, _, _ = find_top_element(X_new, y_mean, y_mean, nb_new_data, verbose)
    print("For exploration:")
    explore_top, _, _ = find_top_element(X_new,y_mean, y_stdv, nb_new_data, verbose)

    return ucb_top, ratio, exploit_top, explore_top


def probability_of_improvement(mu, sigma, current_best):
    """
    Calculate Probability of Improvement (PI) for Gaussian process predictions.

    Parameters:
    - mu: Mean of the Gaussian process prediction.
    - sigma: Standard deviation of the Gaussian process prediction.
    - current_best: Current best observed value.
    Returns:
    - pi: Probability of Improvement.
    """

    # Avoid division by zero
    sigma = sigma + 1e-4

    # Calculate standard normal cumulative distribution function
    z = (mu - current_best) / sigma
    pi = norm.cdf(z)

    return pi


def expected_improvement(mu, sigma, current_best):
    """
    Calculate Expected Improvement (EI) for Gaussian process predictions.

    Parameters:
    - mu: Mean of the Gaussian process prediction.
    - sigma: Standard deviation of the Gaussian process prediction.
    - current_best: Current best observed value.
    - epsilon: Small positive constant to avoid division by zero.

    Returns:
    - ei: Expected Improvement.
    """

    # Avoid division by zero
    sigma = sigma + 1e-4

    # Calculate standard normal cumulative distribution function
    z = (mu - current_best) / sigma
    ei = (mu - current_best) * norm.cdf(z) + sigma * norm.pdf(z)

    return ei

def upper_confident_bound(mu, sigma, theta, r2):
    """
    Calculate UCB for Gaussian process predictions.

    Parameters:
    - mu: Mean of the Gaussian process prediction.
    - sigma: Standard deviation of the Gaussian process prediction.
    - epsilon: Small positive constant to avoid division by zero.

    Returns:
    - ucb: 
    """
    if r2 <= 0.8:
        ucb = mu + theta*sigma
    else:
        ucb = mu
    return ucb


