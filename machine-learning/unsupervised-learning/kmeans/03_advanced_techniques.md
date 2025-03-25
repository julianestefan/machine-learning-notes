# K-means Clustering: Advanced Techniques

Beyond the basic K-means algorithm, several advanced techniques can improve clustering performance, handle specific data characteristics, or extend the algorithm's capabilities.

## 1. K-means++ Initialization

The standard K-means algorithm initializes centroids randomly, which can lead to poor convergence. K-means++ selects initial centroids that are distant from each other:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate sample data
X, _ = make_blobs(n_samples=500, centers=5, cluster_std=0.9, random_state=42)

# Compare random initialization vs k-means++
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Standard K-means with random initialization
km_random = KMeans(n_clusters=5, init='random', n_init=10, random_state=42)
km_random_labels = km_random.fit_predict(X)
axes[0].scatter(X[:, 0], X[:, 1], c=km_random_labels, cmap='viridis', alpha=0.7)
axes[0].scatter(km_random.cluster_centers_[:, 0], km_random.cluster_centers_[:, 1], 
                c='red', marker='X', s=200)
axes[0].set_title(f'Random Initialization\nInertia: {km_random.inertia_:.2f}')

# K-means with k-means++ initialization (default in scikit-learn)
km_plus = KMeans(n_clusters=5, init='k-means++', n_init=10, random_state=42)
km_plus_labels = km_plus.fit_predict(X)
axes[1].scatter(X[:, 0], X[:, 1], c=km_plus_labels, cmap='viridis', alpha=0.7)
axes[1].scatter(km_plus.cluster_centers_[:, 0], km_plus.cluster_centers_[:, 1], 
                c='red', marker='X', s=200)
axes[1].set_title(f'K-means++ Initialization\nInertia: {km_plus.inertia_:.2f}')

plt.tight_layout()
plt.show()
```

## 2. Mini-Batch K-means

For large datasets, standard K-means can be computationally expensive. Mini-batch K-means processes small random batches of data, significantly improving speed with minimal impact on quality:

```python
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.datasets import make_blobs

# Generate large sample data
X, _ = make_blobs(n_samples=100000, centers=8, random_state=42)

# Compare standard K-means vs Mini-batch K-means
algorithms = [
    ('K-means', KMeans(n_clusters=8, random_state=42)),
    ('Mini-Batch K-means', MiniBatchKMeans(n_clusters=8, batch_size=1000, random_state=42))
]

plt.figure(figsize=(12, 5))
for i, (name, algorithm) in enumerate(algorithms):
    t0 = time.time()
    algorithm.fit(X)
    t1 = time.time()
    
    # Plot results
    plt.subplot(1, 2, i+1)
    plt.scatter(X[:, 0], X[:, 1], c=algorithm.labels_, cmap='viridis', alpha=0.1, s=1)
    plt.scatter(algorithm.cluster_centers_[:, 0], algorithm.cluster_centers_[:, 1], 
                c='red', marker='X', s=100)
    plt.title(f'{name}\nTime: {t1-t0:.2f}s, Inertia: {algorithm.inertia_:.0f}')
    
plt.tight_layout()
plt.show()

print(f"Standard K-means inertia: {algorithms[0][1].inertia_:.0f}")
print(f"Mini-batch K-means inertia: {algorithms[1][1].inertia_:.0f}")
print(f"Speed improvement: {(time.time() - t0) / (t1 - t0):.1f}x faster")
```

## 3. Soft K-means (Fuzzy C-means)

Standard K-means provides hard assignments (a point belongs to exactly one cluster). Fuzzy C-means assigns each point a probability of belonging to each cluster:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import skfuzzy as fuzz

# Generate sample data
X, _ = make_blobs(n_samples=200, centers=4, cluster_std=0.60, random_state=42)

# Apply fuzzy c-means
# Transpose data for skfuzzy, which expects features as first dimension
X_t = X.T
n_clusters = 4

# Apply FCM algorithm
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    X_t, n_clusters, 2, error=0.005, maxiter=1000, init=None
)

# De-transpose the cluster centers
cluster_centers = cntr.T

# Get the highest membership for each point
cluster_membership = np.argmax(u, axis=0)

# Plot the result
plt.figure(figsize=(10, 8))

# Plot points colored by highest probability cluster
plt.scatter(X[:, 0], X[:, 1], c=cluster_membership, cmap='viridis', alpha=0.7)

# Plot cluster centers
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='X', s=200, 
            label='Cluster Centers')

# For selected points, show their probabilities visually
selected_points = [10, 50, 100, 150]  # Indices of points to highlight
for idx in selected_points:
    # Get probabilities
    probs = u[:, idx]
    # Plot point with higher opacity
    plt.scatter(X[idx, 0], X[idx, 1], c='black', s=100, alpha=1, edgecolor='black')
    
    # Add text showing membership probabilities
    text = "\n".join([f"C{i}: {p:.2f}" for i, p in enumerate(probs)])
    plt.annotate(text, 
                 xy=(X[idx, 0], X[idx, 1]),
                 xytext=(X[idx, 0] + 0.1, X[idx, 1] + 0.1),
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

plt.title('Fuzzy C-means Clustering with Membership Probabilities')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()
```

## 4. Hierarchical K-means

Hierarchical K-means combines hierarchical clustering with K-means for improved performance on nested cluster structures:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate nested cluster data
np.random.seed(42)
# Create 3 main clusters
X1, _ = make_blobs(n_samples=300, centers=3, cluster_std=1.5, random_state=42)
# Create sub-clusters within each main cluster
X = np.vstack([
    X1,
    np.random.normal(loc=[-5, -5], scale=0.3, size=(100, 2)),  # Subcluster 1
    np.random.normal(loc=[-4, -4], scale=0.3, size=(100, 2)),  # Subcluster 2
    np.random.normal(loc=[5, 5], scale=0.3, size=(100, 2)),    # Subcluster 3
    np.random.normal(loc=[4, 4], scale=0.3, size=(100, 2)),    # Subcluster 4
])

# Function to perform hierarchical K-means
def hierarchical_kmeans(X, top_level_k=3, second_level_k=2, random_state=None):
    # First level clustering
    kmeans_1 = KMeans(n_clusters=top_level_k, random_state=random_state)
    first_labels = kmeans_1.fit_predict(X)
    
    # Second level clustering within each first-level cluster
    final_labels = np.zeros_like(first_labels)
    centers = []
    
    label_offset = 0
    for i in range(top_level_k):
        # Get points in this cluster
        mask = (first_labels == i)
        subcluster_points = X[mask]
        
        if len(subcluster_points) > second_level_k:  # Only subdivide if enough points
            # Apply K-means to this subcluster
            kmeans_sub = KMeans(n_clusters=second_level_k, random_state=random_state)
            sub_labels = kmeans_sub.fit_predict(subcluster_points)
            
            # Adjust labels to be continuous across all clusters
            final_labels[mask] = sub_labels + label_offset
            label_offset += second_level_k
            
            # Store centers
            centers.extend(kmeans_sub.cluster_centers_)
        else:
            # If too few points, don't subdivide
            final_labels[mask] = label_offset
            label_offset += 1
            centers.append(np.mean(subcluster_points, axis=0))
    
    return final_labels, np.array(centers)

# Apply hierarchical K-means
labels, centers = hierarchical_kmeans(X, top_level_k=3, second_level_k=2, random_state=42)

# Visualize the results
plt.figure(figsize=(12, 6))

# Plot standard K-means with k=6
kmeans_standard = KMeans(n_clusters=6, random_state=42)
standard_labels = kmeans_standard.fit_predict(X)

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=standard_labels, cmap='viridis', alpha=0.7)
plt.scatter(kmeans_standard.cluster_centers_[:, 0], kmeans_standard.cluster_centers_[:, 1], 
            c='red', marker='X', s=200)
plt.title('Standard K-means (k=6)')

# Plot hierarchical K-means
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.7)
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200)
plt.title('Hierarchical K-means (3 main clusters → 2 subclusters)')

plt.tight_layout()
plt.show()
```

## 5. Kernel K-means

Kernel K-means applies a kernel transformation to handle non-linearly separable clusters:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import KernelCenterer

# Generate concentric rings dataset
np.random.seed(42)
n_samples = 500
# First ring
r1 = 2 * np.random.rand(n_samples // 2, 1)
theta1 = 2 * np.pi * np.random.rand(n_samples // 2, 1)
X1 = np.hstack([r1 * np.cos(theta1), r1 * np.sin(theta1)])
# Second ring
r2 = 3 + np.random.rand(n_samples // 2, 1)
theta2 = 2 * np.pi * np.random.rand(n_samples // 2, 1)
X2 = np.hstack([r2 * np.cos(theta2), r2 * np.sin(theta2)])
X = np.vstack([X1, X2])
y_true = np.hstack([np.zeros(n_samples // 2), np.ones(n_samples // 2)])

# Function to perform approximate kernel K-means using RBF kernel
def kernel_kmeans(X, n_clusters, gamma=1.0, max_iter=100):
    n_samples = X.shape[0]
    
    # Compute the kernel matrix
    K = rbf_kernel(X, gamma=gamma)
    
    # Initialize random cluster assignments
    labels = np.random.randint(0, n_clusters, size=n_samples)
    
    for _ in range(max_iter):
        old_labels = labels.copy()
        
        # For each cluster
        for i in range(n_clusters):
            # Compute cluster size
            cluster_size = np.sum(labels == i)
            if cluster_size == 0:
                continue
                
            # Compute in-cluster average kernel similarities for points in cluster i
            mask = (labels == i)
            K_cluster = K[:, mask]
            sim_in_cluster = np.sum(K_cluster, axis=1) / cluster_size
            
            # Compute average similarity within the cluster
            K_within = K[mask][:, mask]
            sim_within = np.sum(K_within) / (cluster_size * cluster_size)
            
            # Compute distance to cluster i (in feature space)
            dist_to_i = K.diagonal() - 2 * sim_in_cluster + sim_within
            
            # Update labels based on minimum distance
            labels[dist_to_i < 0] = i
            
        # Check if converged
        if np.all(labels == old_labels):
            break
            
    return labels

# Apply standard K-means
kmeans = KMeans(n_clusters=2, random_state=42)
km_labels = kmeans.fit_predict(X)

# Apply kernel K-means
kernel_labels = kernel_kmeans(X, n_clusters=2, gamma=0.5)

# Visualize the results
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=km_labels, cmap='viridis', alpha=0.7)
plt.title('Standard K-means')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=kernel_labels, cmap='viridis', alpha=0.7)
plt.title('Kernel K-means with RBF Kernel')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()
```

## Comparison of K-means Variants

| Variant | Pros | Cons | Best Use Case |
|---------|------|------|---------------|
| **Standard K-means** | Simple, fast, easy to implement | Sensitive to initialization, limited to convex clusters | General-purpose clustering with well-separated, globular clusters |
| **K-means++** | Better initialization, often better results | Slightly slower initialization | Almost always preferable to standard random initialization |
| **Mini-Batch K-means** | Much faster, scales to large datasets | Slightly less accurate | Very large datasets where speed is critical |
| **Fuzzy C-means** | Soft assignments, uncertainty measurement | More parameters to tune, slower | When points may belong to multiple clusters |
| **Hierarchical K-means** | Can discover nested structures | More complex implementation | Nested or hierarchical cluster structures |
| **Kernel K-means** | Can find non-linearly separable clusters | Computationally expensive for large datasets | Non-globular clusters (e.g., concentric circles) |

## Advanced Implementation Tips

1. **Parallel processing**: Use `n_jobs=-1` in scikit-learn's KMeans for multi-core processing
2. **Dimension reduction**: Apply PCA before K-means for high-dimensional data
3. **Ensemble clustering**: Combine multiple K-means runs for more robust results
4. **Online learning**: For streaming data, use incremental K-means approaches
5. **Custom distance metrics**: For specialized applications, implement custom distance functions 