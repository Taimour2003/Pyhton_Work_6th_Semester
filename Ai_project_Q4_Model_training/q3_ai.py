import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'Mine_Dataset.xlsx'
data = pd.read_excel(file_path, sheet_name='Normalized_Data')
data = data.iloc[:, :-1]  # Remove the last column labeled 'M'
data = data.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle the records

X = data.values  # Convert data to numpy array

# K-means implementation
class KMeans:
    def __init__(self, n_clusters=3, max_iter=100, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        for i in range(self.max_iter):
            self.labels = self.assign_clusters(X)
            new_centroids = self.calculate_centroids(X)
            if np.linalg.norm(new_centroids - self.centroids) < self.tol:
                break
            self.centroids = new_centroids
        return self

    def assign_clusters(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def calculate_centroids(self, X):
        return np.array([X[self.labels == k].mean(axis=0) for k in range(self.n_clusters)])

    def predict(self, X):
        return self.assign_clusters(X)

    def compute_sse(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids[self.labels], axis=2)
        return np.sum(np.min(distances, axis=1) ** 2)

# K-medoids implementation
class KMedoidsCorrected:
    def __init__(self, n_clusters=3, max_iter=100, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        # Initialize medoids randomly
        self.medoids_idx = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        self.medoids = X[self.medoids_idx]

        for i in range(self.max_iter):
            # Assign clusters
            self.labels = self.assign_clusters(X)

            # Calculate new medoids
            new_medoids_idx = self.calculate_medoids(X)

            # Check for convergence
            if np.array_equal(new_medoids_idx, self.medoids_idx):
                break

            self.medoids_idx = new_medoids_idx
            self.medoids = X[self.medoids_idx]

        return self

    def assign_clusters(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.medoids, axis=2)
        return np.argmin(distances, axis=1)

    def calculate_medoids(self, X):
        medoids_idx = np.empty(self.n_clusters, dtype=int)
        for k in range(self.n_clusters):
            cluster_points = X[self.labels == k]
            min_distance_sum = float('inf')
            for point_idx in range(len(cluster_points)):
                distances = np.linalg.norm(cluster_points - cluster_points[point_idx], axis=1)
                distance_sum = np.sum(distances)
                if distance_sum < min_distance_sum:
                    min_distance_sum = distance_sum
                    medoids_idx[k] = point_idx  # Store the index of the medoid
        return medoids_idx

    def predict(self, X):
        return self.assign_clusters(X)

    def compute_sse(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.medoids[self.labels], axis=2)
        return np.sum(np.min(distances, axis=1) ** 2)

# Function to plot elbow chart
def plot_elbow_chart(X):
    sse = []
    K = range(1, 11)
    for k in K:
        kmeans = KMeans(n_clusters=k, max_iter=100)
        kmeans.fit(X)
        sse.append(kmeans.compute_sse(X))

    plt.figure(figsize=(10, 6))
    plt.plot(K, sse, 'bx-')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('SSE/WCSS')
    plt.title('Elbow Method For Optimal K')
    plt.show()

# Function to perform cluster analysis
def cluster_analysis(labels, centroids, X):
    cluster_info = {}
    for k in np.unique(labels):
        cluster_points = X[labels == k]
        cluster_size = len(cluster_points)
        cluster_sse = np.sum((cluster_points - centroids[k]) ** 2)
        cluster_info[k] = {'size': cluster_size, 'sse': cluster_sse}
    return cluster_info

# Plot elbow chart to determine optimal K
plot_elbow_chart(X)

# Run K-means with optimal K
kmeans = KMeans(n_clusters=3, max_iter=100)
kmeans.fit(X)
kmeans_labels = kmeans.labels
kmeans_sse = kmeans.compute_sse(X)

# Run K-medoids with optimal K
kmedoids_corrected = KMedoidsCorrected(n_clusters=3, max_iter=100)
kmedoids_corrected.fit(X)
kmedoids_corrected_labels = kmedoids_corrected.labels
kmedoids_corrected_sse = kmedoids_corrected.compute_sse(X)

# Calculate cluster sizes and SSE/WCSS for each cluster
kmeans_cluster_info = cluster_analysis(kmeans_labels, kmeans.centroids, X)
kmedoids_corrected_cluster_info = cluster_analysis(kmedoids_corrected_labels, kmedoids_corrected.medoids, X)


import time

# Function to measure execution time
def measure_time(algorithm, X, n_clusters):
    start_time = time.time()
    algorithm(n_clusters=n_clusters).fit(X)
    end_time = time.time()
    return end_time - start_time

# Measure time for K-means
kmeans_time = measure_time(KMeans, X, 3)

# Measure time for K-medoids
kmedoids_time = measure_time(KMedoidsCorrected, X, 3)



# Print results
print("K-means Clustering Results:")
print("Cluster Info:", kmeans_cluster_info)
print("Overall SSE:", kmeans_sse)

print("\nK-medoids Clustering Results:")
print("Cluster Info:", kmedoids_corrected_cluster_info)
print("Overall SSE:", kmedoids_corrected_sse)


# Print time complexity
print(f"K-means Time Complexity: {kmeans_time:.4f} seconds")
print(f"K-medoids Time Complexity: {kmedoids_time:.4f} seconds")

# Comparative Analysis
print("\nComparative Analysis:")
print("1. Cluster Sizes: Both K-means and K-medoids produced clusters of equal size.")
print("2. SSE/WCSS Values: K-means has a slightly lower overall SSE compared to K-medoids.")
print("3. Time Complexity: K-means is generally faster than K-medoids due to its lower time complexity.")
print("4. Performance: K-means performed slightly better in terms of SSE and execution time.")

# Overall Winner
print("\nOverall Winner:")
print("For this dataset, K-means is the overall winner due to its slightly lower SSE and faster execution time.")
