# K-Means Clustering: Data Preprocessing Techniques

When performing K-means clustering, properly preprocessing your data is crucial for obtaining meaningful clusters. Features with different scales or large variances can dominate the distance calculations, leading to poor clustering results.

## Common Preprocessing Methods for K-means

| Method | Purpose | When to Use | Effect |
|--------|---------|-------------|--------|
| **StandardScaler** | Standardizes features by removing the mean and scaling to unit variance | When features have different scales and distributions | Each feature will have mean=0 and std=1 |
| **Normalizer** | Scales each sample to unit norm | When only the direction (not magnitude) of the samples matters | Each sample will have norm=1 |
| **MinMaxScaler** | Scales features to a given range (usually [0,1]) | When you need bounded values within a specific range | All features will be between 0 and 1 |
| **RobustScaler** | Uses statistics that are robust to outliers | When your data contains significant outliers | Scales based on median and quantiles |

## StandardScaler Example

```python
# Perform the necessary imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Example data (replace with your actual data)
# samples could be a dataset with features of different scales
# species could be ground truth labels (if available)
samples = np.random.rand(100, 2) * np.array([10, 1])  # Features with different scales
species = ['A', 'B', 'C', 'D'] * 25  # Optional: Ground truth for evaluation

# Create scaler: scaler
scaler = StandardScaler()

# Create KMeans instance: kmeans
kmeans = KMeans(n_clusters=4, random_state=42)

# Create pipeline: pipeline
# Using a pipeline ensures that the scaling is applied correctly during both fit and predict
pipeline = make_pipeline(scaler, kmeans)

# Fit the pipeline to samples
pipeline.fit(samples)

# Calculate the cluster labels: labels
labels = pipeline.predict(samples)

# Create a DataFrame with labels and species as columns: df
df = pd.DataFrame({'labels': labels, 'species': species})

# Create crosstab to evaluate clustering against ground truth (if available)
ct = pd.crosstab(df["labels"], df["species"])
print("Cluster distribution:")
print(ct)

# Visualize the clusters
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.scatter(samples[:, 0], samples[:, 1], c='gray', alpha=0.5)
plt.title("Original Data (Unscaled)")
plt.xlabel("Feature 1 (Large Scale)")
plt.ylabel("Feature 2 (Small Scale)")

plt.subplot(1, 2, 2)
plt.scatter(samples[:, 0], samples[:, 1], c=labels, cmap='viridis', alpha=0.7)
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    c='red', 
    marker='X', 
    s=100
)
plt.title("K-means with StandardScaler")
plt.xlabel("Feature 1 (Standardized)")
plt.ylabel("Feature 2 (Standardized)")
plt.tight_layout()
```

StandardScaler standardizes features by removing the mean and scaling to unit variance. This is especially useful when your features have significantly different scales or distributions.

## Normalizer Example

The Normalizer rescales each sample (row) independently to have unit norm:

```python 
# Imports
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Example: financial data (daily stock price movements)
# Replace with your actual data
np.random.seed(42)
movements = np.random.randn(20, 5)  # 20 companies, 5 days of price changes
companies = [f'Company_{i}' for i in range(1, 21)]

# Create a normalizer: normalizer
normalizer = Normalizer()

# Create a KMeans model with 4 clusters: kmeans
kmeans = KMeans(n_clusters=4, random_state=42)

# Make a pipeline chaining normalizer and kmeans: pipeline
pipeline = make_pipeline(normalizer, kmeans)

# Fit pipeline to the daily price movements
pipeline.fit(movements)

# Predict the cluster labels: labels
labels = pipeline.predict(movements)

# Create a DataFrame aligning labels and companies: df
df = pd.DataFrame({'labels': labels, 'companies': companies})

# Display df sorted by cluster label
print("Companies grouped by similar price movement patterns:")
print(df.sort_values("labels"))

# Visualize the normalized data (if 2D or reducible to 2D)
# For high-dimensional data, you might want to use techniques like PCA or t-SNE to visualize
```

## When to Use Each Scaler

- **StandardScaler**: Best for features with different units or when assuming normally distributed data
- **Normalizer**: Useful for text data or when the direction of the data matters more than the magnitude
- **MinMaxScaler**: Good for image processing or when you need bounded values
- **RobustScaler**: Effective when dealing with datasets containing outliers

## Best Practices for Preprocessing in K-means

1. Always scale your data before applying K-means
2. Use a pipeline to ensure consistent preprocessing between fit and predict steps
3. Try different preprocessing methods to see which works best for your specific dataset
4. Visualize your data before and after preprocessing to gain insights
5. For high-dimensional data, consider dimensionality reduction techniques (PCA, t-SNE) before clustering 