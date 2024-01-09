import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split


# Generate synthetic data with 11 features
np.random.seed(42)
X = np.random.rand(1000, 11)  # 11 features
# Create a polynomial relationship with added noise
coefficients = np.random.rand(11)
y = np.dot(X**2, coefficients) + 0.1 * np.random.randn(1000)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and train an MLPRegressor with 11 input features
mlp = MLPRegressor(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)
mlp.fit(X_train, y_train)

# Define the black-box function using the trained MLP
def objective2(input_matrix):
    # Ensure input_matrix is a 2D numpy array
    input_data = np.array(input_matrix)
    
    # Predict using the trained MLP
    predictions = mlp.predict(input_data)
    
    return predictions

# objective function
def objective(features, seed = None):
    if isinstance(features, pd.DataFrame):
        features = features.to_numpy()

    elif not isinstance(features, np.ndarray):
        raise ValueError("Input must be either a NumPy array or a Pandas DataFrame")
    
    # Define a more complex relationship between features and Y
    Y = (
        2 * features[:, 0] +
        0.01 * features[:, 1] ** 2 +
        3 *np.sin(features[:, 2]) 
        - 2 * np.exp(-features[:, 3]) +
        0.5 * features[:, 4] * features[:, 5] 
        - features[:, 6] ** 3 +
        1 * np.cos(features[:, 7]) +
        0.05 * features[:, 8] ** 2 
        - 0.1 * np.tan(features[:, 9])
    )
    # Add some random noise to the target variable
    np.random.seed(seed)
    noise = np.random.normal(0, 0.5, features.shape[0])
    Y += noise
    # Add constant
    Y = Y + 5

    return Y

def objective3(X):

    X = np.array(X)
    
    np.random.seed(31)
    coefficients = np.random.rand(11)
    y = (
        np.sin(np.sum(X, axis=1) * 3) +  # Sine function
        np.dot(X**2, coefficients) +      # Quadratic terms
        0.1 * np.random.randn(len(X))        # Noise
    )
    return y