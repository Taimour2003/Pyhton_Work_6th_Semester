import pandas as pd
import numpy as np
import time
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn_extra.cluster import KMedoids as SklearnKMedoids
from sklearn.metrics import pairwise_distances

# Load the dataset
file_path = 'Mine_Dataset.xlsx'
data = pd.read_excel(file_path, sheet_name='Normalized_Data')
data = data.iloc[:, :-1]  # Remove the last column labeled 'M'
data = data.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle the records

# Define X
X = data.values  # Convert data to numpy array

# Function to calculate cluster sizes and SSE/WCSS for each cluster
def cluster_analysis(labels, centroids, X):
    cluster_info = {}
    for k in np.unique(labels):
        cluster_points = X[labels == k]
        cluster_size = len(cluster_points)
        cluster_sse = np.sum((cluster_points - centroids[k]) ** 2)
        cluster_info[k] = {'size': cluster_size, 'sse': cluster_sse}
    return cluster_info

# Function to measure execution time
def measure_time_sklearn(algorithm, X, n_clusters):
    start_time = time.time()
    model = algorithm(n_clusters=n_clusters, random_state=42).fit(X)
    end_time = time.time()
    return end_time - start_time, model

# Sklearn K-means
kmeans_time_sklearn, kmeans_model_sklearn = measure_time_sklearn(SklearnKMeans, X, 3)
kmeans_labels_sklearn = kmeans_model_sklearn.labels_
kmeans_sse_sklearn = kmeans_model_sklearn.inertia_
kmeans_iterations_sklearn = kmeans_model_sklearn.n_iter_

# Sklearn K-medoids
kmedoids_time_sklearn, kmedoids_model_sklearn = measure_time_sklearn(SklearnKMedoids, X, 3)
kmedoids_labels_sklearn = kmedoids_model_sklearn.labels_
kmedoids_sse_sklearn = sum(np.min(pairwise_distances(X, kmedoids_model_sklearn.cluster_centers_), axis=1) ** 2)
kmedoids_iterations_sklearn = kmedoids_model_sklearn.n_iter_

# Calculate cluster sizes for sklearn K-means
kmeans_cluster_info_sklearn = cluster_analysis(kmeans_labels_sklearn, kmeans_model_sklearn.cluster_centers_, X)

# Calculate cluster sizes for sklearn K-medoids
kmedoids_cluster_info_sklearn = cluster_analysis(kmedoids_labels_sklearn, kmedoids_model_sklearn.cluster_centers_, X)

# Print results for sklearn K-means
print("Sklearn K-means Clustering Results:")
print("Iterations:", kmeans_iterations_sklearn)
print("Cluster Info:", kmeans_cluster_info_sklearn)
print("Overall SSE:", kmeans_sse_sklearn)
print("Time Complexity:", kmeans_time_sklearn, "seconds")

# Print results for sklearn K-medoids
print("\nSklearn K-medoids Clustering Results:")
print("Iterations:", kmedoids_iterations_sklearn)
print("Cluster Info:", kmedoids_cluster_info_sklearn)
print("Overall SSE:", kmedoids_sse_sklearn)
print("Time Complexity:", kmedoids_time_sklearn, "seconds")

# Comparative Analysis
print("\nComparative Analysis:")
print("1. Iterations: The number of iterations may vary due to differences in implementation details.")
print("2. Cluster Sizes: Both custom and sklearn implementations should produce similar cluster sizes.")
print("3. SSE/WCSS Values: The SSE values should be comparable if the implementations are correct.")
print("4. Time Complexity: Sklearn implementations are generally optimized and may run faster.")
print("5. Alignment: If results align closely, it indicates correctness. Differences may arise from optimization techniques in sklearn.")
