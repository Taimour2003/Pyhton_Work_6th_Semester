# Importing required libraries
from sklearn.datasets import fetch_california_housing  # Dataset loader for California housing data
import pandas as pd  # For data manipulation and analysis
import matplotlib.pyplot as plt  # For plotting
import seaborn as sns  # For advanced visualization

# Load the California Housing dataset
data = fetch_california_housing()

# Convert the dataset into a pandas DataFrame with appropriate column names
df = pd.DataFrame(data.data, columns=data.feature_names)

# Add the target column (house value) to the DataFrame
df['Target'] = data.target

# Display the first 5 rows of the dataset to understand its structure
print(df.head())

# Get statistical summary of the dataset (mean, std, min, max, etc.)
print(df.describe())

# Visualize pairwise relationships among selected features and the target
sns.pairplot(df[['MedInc', 'AveRooms', 'AveOccup', 'Latitude', 'Longitude', 'Target']])
plt.show()


# ---------------------------- DATA PREPROCESSING ---------------------------- #

from sklearn.model_selection import train_test_split  # For splitting data into training and testing sets
from sklearn.preprocessing import StandardScaler  # For standardizing features (important for k-NN)

# Separate features (X) and target (y)
X = df.drop('Target', axis=1)  # Features (independent variables)
y = df['Target']  # Target (dependent variable - house value)

# Split the dataset into training (80%) and testing (20%) subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the feature values for better model performance (especially for distance-based models like k-NN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit on training data and transform it
X_test_scaled = scaler.transform(X_test)  # Only transform the test data (no fitting)


# ---------------------------- MODEL TRAINING ---------------------------- #

from sklearn.linear_model import LinearRegression  # Linear Regression model
from sklearn.neighbors import KNeighborsRegressor  # k-NN Regressor

# Train a Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)  # Fit the model on scaled training data

# Train multiple k-NN models with different values of k
knn_models = {}  # Dictionary to store k-NN models
for k in [3, 5, 7]:  # Trying different neighbor sizes
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)  # Fit the k-NN model on training data
    knn_models[k] = knn  # Store the trained model in dictionary


# ---------------------------- MODEL EVALUATION ---------------------------- #

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error  # Evaluation metrics

# Define a function to evaluate and print performance of a model
def evaluate_model(name, model):
    preds = model.predict(X_test_scaled)  # Predict target values for test set
    print(f"{name}:\n R2: {r2_score(y_test, preds):.3f}, MAE: {mean_absolute_error(y_test, preds):.3f}, MSE: {mean_squared_error(y_test, preds):.3f}\n")

# Evaluate the Linear Regression model
evaluate_model("Linear Regression", lr_model)

# Evaluate each of the k-NN models
for k, model in knn_models.items():
    evaluate_model(f"k-NN (k={k})", model)


# ---------------------------- RESULT COMPARISON ---------------------------- #

import numpy as np  # For numerical operations (used by pandas under the hood too)

# Create a dictionary to collect model performance metrics
results = {
    'Model': [],  # Names of models
    'R2': [],     # R-squared (explained variance)
    'MAE': [],    # Mean Absolute Error
    'MSE': []     # Mean Squared Error
}

# Loop through each model, compute predictions, and store metrics
for label, model in [('Linear', lr_model)] + [(f'kNN_{k}', knn_models[k]) for k in knn_models]:
    preds = model.predict(X_test_scaled)
    results['Model'].append(label)
    results['R2'].append(r2_score(y_test, preds))
    results['MAE'].append(mean_absolute_error(y_test, preds))
    results['MSE'].append(mean_squared_error(y_test, preds))

# Convert the results dictionary to a DataFrame for tabular display
result_df = pd.DataFrame(results)
print(result_df)  # Print the final comparison of model performances
