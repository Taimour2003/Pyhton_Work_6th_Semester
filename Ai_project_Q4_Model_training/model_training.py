from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
# Load the dataset
data = fetch_california_housing()

# Convert to DataFrame for easier exploration
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Target'] = data.target  # Add the target column (median house value)

# Display first 5 rows
print(df.head())
print(df.describe())

# 2. Check for missing values
print("\n* Missing values:")
print(df.isnull().sum())

# # 3. Correlation matrix
# print("\n📌 Correlation with Target:")
# print(df.corr(numeric_only=True)['Target'].sort_values(ascending=False))

plt.figure(figsize=(8, 4))
sns.histplot(df['Target'], bins=40, kde=True, color='green')
plt.title('Distribution of House Prices')
plt.xlabel('Median House Value ($100,000s)')
plt.ylabel('Frequency')
plt.show()

# #correlation matrix plot
# plt.figure(figsize=(10, 8))
# sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
# plt.title("Feature Correlation Heatmap")
# plt.show()


# sns.scatterplot(x='MedInc', y='Target', data=df)
# plt.title('Median Income vs House Price')
# plt.xlabel('Median Income')
# plt.ylabel('House Value')
# plt.show()

# sns.scatterplot(x='AveRooms', y='Target', data=df)
# plt.title('Average Rooms vs House Price')
# plt.xlabel('Average Rooms')
# plt.ylabel('House Value')
# plt.show()


# df['AgeGroup'] = pd.cut(df['HouseAge'], bins=[0, 20, 35, 50, 60], labels=['0-20', '21-35', '36-50', '51+'])

# sns.boxplot(x='AgeGroup', y='Target', data=df)
# plt.title('House Price by Age Group')
# plt.xlabel('House Age Group')
# plt.ylabel('Median House Value')
# plt.show()



# Features and Target
X = df.drop('Target', axis=1)  # All columns except target
y = df['Target']               # Target column (house prices)

# Split into training and testing (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training data: {X_train.shape}, Testing data: {X_test.shape}")


# Create scaler
scaler = StandardScaler()

# Fit on training data and transform both train & test
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


def evaluate_model(name, y_true, y_pred):
    print(f"\n {name}")
    print("R² Score       :", round(r2_score(y_true, y_pred), 4))
    print("MAE            :", round(mean_absolute_error(y_true, y_pred), 4))
    print("MSE            :", round(mean_squared_error(y_true, y_pred), 4))
    print("RMSE           :", round(root_mean_squared_error(y_true, y_pred), 4))
    


# Create and train the model
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

# Predict on test set
y_pred_lr = lr_model.predict(X_test_scaled)

# Evaluation
print("\n Linear Regression:")
evaluate_model("Linear Regression", y_test, y_pred_lr)



# Linear Regression

# k-NN Models
results = []

for k in [3, 5, 7]:
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred_knn = knn.predict(X_test_scaled)
    results.append({
        "Model": f"k-NN (k={k})",
        "R2": r2_score(y_test, y_pred_knn),
        "MAE": mean_absolute_error(y_test, y_pred_knn),
        "MSE": mean_squared_error(y_test, y_pred_knn),
        "RMSE": root_mean_squared_error(y_test, y_pred_knn)
    })

# Add Linear Regression
results.append({
    "Model": "Linear Regression",
    "R2": r2_score(y_test, y_pred_lr),
    "MAE": mean_absolute_error(y_test, y_pred_lr),
    "MSE": mean_squared_error(y_test, y_pred_lr),
    "RMSE": root_mean_squared_error(y_test, y_pred_lr)
})

results_df = pd.DataFrame(results)
print(results_df)
