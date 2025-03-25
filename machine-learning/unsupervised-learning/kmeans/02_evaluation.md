# K-means Clustering: Evaluation and Performance Metrics

Evaluating unsupervised learning algorithms like K-means is challenging since there's typically no ground truth for comparison. However, several methods can help assess clustering quality and determine the optimal number of clusters.

## 1. Cross-Tabulation (When Ground Truth is Available)

When you have labeled data (e.g., species, categories), you can evaluate clustering quality by comparing cluster assignments to known labels:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, adjusted_rand_score, normalized_mutual_info_score

# Example data with known labels (replace with your actual data)
np.random.seed(42)
samples = np.vstack([
    np.random.normal(loc=[2, 2], scale=0.5, size=(30, 2)),  # Class A
    np.random.normal(loc=[8, 8], scale=0.5, size=(30, 2)),  # Class B
    np.random.normal(loc=[2, 8], scale=0.5, size=(30, 2))   # Class C
])
varieties = ['A'] * 30 + ['B'] * 30 + ['C'] * 30  # Ground truth labels

# Create a KMeans model with 3 clusters
model = KMeans(n_clusters=3, random_state=42)

# Use fit_predict to fit model and obtain cluster labels
labels = model.fit_predict(samples)

# Create a DataFrame with labels and varieties as columns
df = pd.DataFrame({'labels': labels, 'varieties': varieties})

# Create crosstab for visual inspection
ct = pd.crosstab(df["labels"], df["varieties"])
print("Cross-tabulation of clusters vs. known classes:")
print(ct)

# Calculate evaluation metrics
ari = adjusted_rand_score(varieties, labels)
nmi = normalized_mutual_info_score(varieties, labels)
print(f"Adjusted Rand Index: {ari:.3f}")
print(f"Normalized Mutual Information: {nmi:.3f}")
```

## 2. Inertia (Within-Cluster Sum of Squares)

Inertia measures how internally coherent clusters are. Lower values indicate better clustering:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate sample data with 4 clusters
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)

# Calculate inertia for different numbers of clusters
ks = range(1, 10)  # Try clusters from 1 to 9
inertias = []

for k in ks:
    # Create a KMeans instance with k clusters
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    
    # Fit model to samples
    model.fit(X)
    
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    
# Plot ks vs inertias - the "Elbow Method"
plt.figure(figsize=(10, 6))
plt.plot(ks, inertias, 'o-', color='blue')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.xticks(ks)
plt.grid(True, linestyle='--', alpha=0.7)

# Add an annotation for the "elbow" point (in this example, k=4)
# In real applications, you would determine this programmatically or visually
plt.annotate('Elbow point',
             xy=(4, inertias[3]),
             xytext=(5, inertias[3] + 20),
             arrowprops=dict(facecolor='red', shrink=0.05),
             )
            
plt.show()
```

## 3. Silhouette Score

The silhouette score measures how similar points are to their own cluster compared to other clusters. Higher values (closer to 1) indicate better clustering:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.datasets import make_blobs

# Generate sample data
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)

# Calculate silhouette scores for different numbers of clusters
ks = range(2, 10)  # Silhouette score requires at least 2 clusters
silhouette_scores = []

for k in ks:
    # Create a KMeans instance with k clusters
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    
    # Get cluster assignments
    cluster_labels = model.fit_predict(X)
    
    # Calculate silhouette score
    silhouette_avg = silhouette_score(X, cluster_labels)
    silhouette_scores.append(silhouette_avg)
    print(f"For k={k}, silhouette score: {silhouette_avg:.3f}")
    
# Plot silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(ks, silhouette_scores, 'o-', color='green')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Different Cluster Counts')
plt.xticks(ks)
plt.grid(True, linestyle='--', alpha=0.7)

# Find and annotate optimal k (highest silhouette score)
optimal_k = ks[np.argmax(silhouette_scores)]
plt.annotate(f'Optimal k={optimal_k}',
             xy=(optimal_k, max(silhouette_scores)),
             xytext=(optimal_k + 1, max(silhouette_scores) - 0.02),
             arrowprops=dict(facecolor='red', shrink=0.05),
             )
            
plt.show()
```

## 4. Visualizing Cluster Quality

For 2D data, visualization can provide intuitive validation of clustering quality:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate sample data
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)

# Create a KMeans instance with 4 clusters
model = KMeans(n_clusters=4, random_state=42)

# Get cluster assignments
cluster_labels = model.fit_predict(X)

# Plot the clusters
plt.figure(figsize=(10, 8))
plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap='viridis', s=50, alpha=0.8)
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], 
            c='red', marker='X', s=200, label='Centroids')

# Draw circles around each centroid showing inertia
for i, center in enumerate(model.cluster_centers_):
    # Get points in this cluster
    cluster_points = X[cluster_labels == i]
    
    # Calculate average distance to centroid (approximation of inertia contribution)
    avg_distance = np.mean(np.linalg.norm(cluster_points - center, axis=1))
    
    # Draw circle
    circle = plt.Circle(center, avg_distance, fill=False, linestyle='--', 
                        edgecolor='gray', alpha=0.5)
    plt.gca().add_patch(circle)
    
plt.title('K-means Clustering with Centroids and Average Distances')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()
```

## 5. Determining the Optimal Number of Clusters

The "elbow method" and silhouette analysis are common approaches, but you can also use:

| Method | Description | Implementation |
|--------|-------------|----------------|
| **Gap Statistic** | Compares inertia to that of random data | sklearn-contrib-py-earth |
| **Davies-Bouldin Index** | Ratio of within-cluster to between-cluster distances | sklearn.metrics.davies_bouldin_score |
| **Calinski-Harabasz Index** | Ratio of between-cluster to within-cluster dispersion | sklearn.metrics.calinski_harabasz_score |
| **Bayesian Information Criterion** | Penalizes model complexity | sklearn.mixture BIC criterion (GMM) |

## Best Practices for K-means Evaluation

1. **Use multiple evaluation methods**: Different metrics may suggest different optimal k values
2. **Consider domain knowledge**: The "right" number of clusters should make sense for your application
3. **Visualize results**: Plot data and clusters whenever possible to validate clustering quality
4. **Run multiple initializations**: K-means is sensitive to initialization; use n_init parameter to run multiple times
5. **Compare to random baselines**: Good clustering should be significantly better than random assignment 