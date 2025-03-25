# Introduction to K-means Clustering

K-means clustering is one of the most popular and straightforward unsupervised machine learning algorithms. It partitions data into K distinct non-overlapping clusters based on feature similarity.

## How K-means Works

1. **Initialization**: Randomly select K points as initial centroids
2. **Assignment**: Assign each data point to the nearest centroid, forming K clusters
3. **Update**: Recalculate the centroids of the new clusters
4. **Repeat**: Iterate steps 2-3 until convergence (centroids no longer move significantly)

## Basic Implementation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Generate example data
np.random.seed(42)
points = np.vstack([
    np.random.normal(loc=[2, 2], scale=0.5, size=(100, 2)),  # Cluster 1
    np.random.normal(loc=[7, 7], scale=0.5, size=(100, 2)),  # Cluster 2
    np.random.normal(loc=[2, 7], scale=0.5, size=(100, 2))   # Cluster 3
])

# Create a KMeans instance with 3 clusters
model = KMeans(n_clusters=3, random_state=42, n_init=10)

# Fit model to points
model.fit(points)

# Get cluster labels for some new points
new_points = np.random.rand(10, 2) * 10
labels = model.predict(new_points)

# Print cluster labels of new_points
print("Cluster assignments for new points:")
print(labels)
```

## Visualizing K-means Clusters and Centroids

Visualization is crucial for understanding clustering results:

```python
# Import pyplot
import matplotlib.pyplot as plt

# Visualize the original data
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(points[:, 0], points[:, 1], alpha=0.5)
plt.title("Original Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

# Visualize the clusters
plt.subplot(1, 2, 2)
# Get cluster labels for the original points
cluster_labels = model.predict(points)

# Assign coordinates for plotting
xs = points[:, 0]
ys = points[:, 1]

# Make a scatter plot of xs and ys, using labels to define the colors
plt.scatter(xs, ys, c=cluster_labels, cmap='viridis', alpha=0.7)

# Get the cluster centers
centroids = model.cluster_centers_

# Coordinates of centroids
centroids_x = centroids[:, 0]
centroids_y = centroids[:, 1]

# Plot the centroids
plt.scatter(centroids_x, centroids_y, marker="X", s=200, c='red', label='Centroids')
plt.title("K-means Clustering (k=3)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.tight_layout()
plt.show()
```

## K-means Applications

- **Customer Segmentation**: Group customers based on purchasing behavior
- **Image Compression**: Reduce color palette to k colors
- **Document Clustering**: Group similar documents for topic modeling
- **Anomaly Detection**: Identify unusual data points far from centroids

## Advantages and Limitations

| Advantages | Limitations |
|------------|-------------|
| Simple and easy to implement | Requires number of clusters (K) to be specified |
| Scales well to large datasets | Sensitive to initial centroid placement |
| Works with numeric data | Cannot handle non-globular clusters |
| Finds globular clusters efficiently | Sensitive to outliers |
| Fast convergence | May converge to local optima |

In the next sections, we'll explore data preprocessing for K-means, evaluate clustering performance, and discuss advanced techniques for determining the optimal number of clusters. 