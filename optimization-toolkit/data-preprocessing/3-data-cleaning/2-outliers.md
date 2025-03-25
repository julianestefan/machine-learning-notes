# Handling Outliers

Outliers are data points that differ significantly from other observations and can have a disproportionate impact on model performance.

## What are Outliers?

Outliers are data points that fall far from the majority of the data. They can be:
- **Valid outliers**: Unusual but legitimate values
- **Measurement errors**: Values resulting from equipment malfunction or human error
- **Processing errors**: Values resulting from data processing mistakes
- **Sampling errors**: Values that don't belong to the population of interest

## Detection Methods

### Visual Methods

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('dataset.csv')

# Select numerical columns
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

# Box plots for outlier detection
plt.figure(figsize=(15, 10))
for i, col in enumerate(numeric_cols[:9], 1):  # Limit to 9 columns for readability
    plt.subplot(3, 3, i)
    sns.boxplot(y=df[col])
    plt.title(f'Box Plot of {col}')
plt.tight_layout()
plt.show()

# Histograms
plt.figure(figsize=(15, 10))
for i, col in enumerate(numeric_cols[:9], 1):
    plt.subplot(3, 3, i)
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()

# Scatter plots (for examining relationships)
if len(numeric_cols) >= 2:
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=df[numeric_cols[0]], y=df[numeric_cols[1]])
    plt.title(f'Scatter Plot: {numeric_cols[0]} vs {numeric_cols[1]}')
    plt.tight_layout()
    plt.show()
```

### Statistical Methods

#### Z-score Method

Identifies outliers based on standard deviations from the mean:

```python
from scipy import stats

# Calculate z-scores
z_scores = stats.zscore(df[numeric_cols])

# Define threshold (typically 3)
threshold = 3

# Find outliers (abs(z-score) > threshold)
outliers_z = (np.abs(z_scores) > threshold)

# Count outliers in each column
outliers_z_count = outliers_z.sum(axis=0)
print("Outliers detected with Z-score method:")
for col, count in zip(numeric_cols, outliers_z_count):
    print(f"{col}: {count} outliers")

# Get outlier rows
outlier_rows = df[outliers_z.any(axis=1)]
print(f"\nTotal rows with outliers: {len(outlier_rows)}")
```

#### IQR Method

Identifies outliers based on the Interquartile Range:

```python
# Calculate Q1, Q3, and IQR
Q1 = df[numeric_cols].quantile(0.25)
Q3 = df[numeric_cols].quantile(0.75)
IQR = Q3 - Q1

# Define bounds (typically 1.5 * IQR)
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Find outliers
outliers_iqr = ((df[numeric_cols] < lower_bound) | (df[numeric_cols] > upper_bound))

# Count outliers in each column
outliers_iqr_count = outliers_iqr.sum(axis=0)
print("Outliers detected with IQR method:")
for col, count in zip(numeric_cols, outliers_iqr_count):
    print(f"{col}: {count} outliers")

# Get outlier rows
outlier_rows_iqr = df[outliers_iqr.any(axis=1)]
print(f"\nTotal rows with outliers: {len(outlier_rows_iqr)}")
```

#### Modified Z-score

More robust for non-normal distributions:

```python
def modified_z_score(data):
    # Calculate median and MAD (Median Absolute Deviation)
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    # Add small constant to avoid division by zero
    mad_e = mad if mad > 0 else 1e-8
    # Calculate modified z-scores
    return 0.6745 * (data - median) / mad_e

# Calculate modified z-scores for each column
for col in numeric_cols:
    df[f'{col}_mod_z'] = modified_z_score(df[col])
    
    # Identify outliers (typically |mod_z| > 3.5)
    outliers = df[np.abs(df[f'{col}_mod_z']) > 3.5]
    print(f"Modified Z-score outliers in {col}: {len(outliers)}")
    
    # Remove temporary column
    df.drop(f'{col}_mod_z', axis=1, inplace=True)
```

### Machine Learning Based Methods

#### Isolation Forest

Identifies outliers based on isolation in a random forest:

```python
from sklearn.ensemble import IsolationForest

# Initialize and fit the model
iso_forest = IsolationForest(contamination=0.05, random_state=42)  # 5% expected outliers
df['outlier_scores'] = iso_forest.fit_predict(df[numeric_cols])

# Outliers are marked as -1
outliers_if = df[df['outlier_scores'] == -1]
print(f"Isolation Forest outliers: {len(outliers_if)}")

# Visualize outliers (for 2D data)
if len(numeric_cols) >= 2:
    plt.figure(figsize=(10, 8))
    plt.scatter(df[numeric_cols[0]], df[numeric_cols[1]], 
                c=df['outlier_scores'], cmap='coolwarm', alpha=0.7)
    plt.colorbar(label='Outlier (-1) vs Inlier (1)')
    plt.title('Isolation Forest Outlier Detection')
    plt.xlabel(numeric_cols[0])
    plt.ylabel(numeric_cols[1])
    plt.tight_layout()
    plt.show()
    
# Remove temporary column
df.drop('outlier_scores', axis=1, inplace=True)
```

#### DBSCAN Clustering

Density-based spatial clustering:

```python
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Standardize data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[numeric_cols])

# Apply DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
df['clusters'] = dbscan.fit_predict(df_scaled)

# Outliers are marked as -1
outliers_dbscan = df[df['clusters'] == -1]
print(f"DBSCAN outliers: {len(outliers_dbscan)}")

# Remove temporary column
df.drop('clusters', axis=1, inplace=True)
```

## Handling Strategies

### 1. Remove Outliers

```python
# Remove outliers based on z-score
df_no_outliers = df.copy()
for col in numeric_cols:
    z = np.abs(stats.zscore(df_no_outliers[col]))
    df_no_outliers = df_no_outliers[z < 3]  # Keep only non-outliers

print(f"Original shape: {df.shape}, After removing outliers: {df_no_outliers.shape}")
```

### 2. Cap Outliers (Winsorizing)

```python
# Function to cap outliers at percentiles
def cap_outliers(df, cols, lower=0.05, upper=0.95):
    df_capped = df.copy()
    for col in cols:
        lower_limit = df[col].quantile(lower)
        upper_limit = df[col].quantile(upper)
        df_capped[col] = df_capped[col].clip(lower=lower_limit, upper=upper_limit)
    return df_capped

# Cap outliers at 5th and 95th percentiles
df_capped = cap_outliers(df, numeric_cols)

# Visualize before and after capping for a selected column
if len(numeric_cols) > 0:
    col = numeric_cols[0]  # Select first numeric column for demonstration
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.boxplot(y=df[col], ax=axes[0])
    axes[0].set_title(f'Original {col}')
    sns.boxplot(y=df_capped[col], ax=axes[1])
    axes[1].set_title(f'Capped {col} (5-95 percentile)')
    plt.tight_layout()
    plt.show()
```

### 3. Transform Features

```python
# Log transformation to reduce outlier impact
for col in numeric_cols:
    # Check if all values are positive
    if (df[col] > 0).all():
        df[f'{col}_log'] = np.log(df[col])
    # For columns with zeros or negative values
    else:
        df[f'{col}_log'] = np.log1p(df[col] - df[col].min() + 1)  # Shift and log1p

# Compare distributions for a selected column
if len(numeric_cols) > 0:
    col = numeric_cols[0]  # Select first numeric column for demonstration
    log_col = f'{col}_log'
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(df[col], kde=True, ax=axes[0])
    axes[0].set_title(f'Original {col} Distribution')
    sns.histplot(df[log_col], kde=True, ax=axes[1])
    axes[1].set_title(f'Log-transformed {col} Distribution')
    plt.tight_layout()
    plt.show()
```

### 4. Use Robust Models

Some models are naturally robust to outliers:

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import HuberRegressor, RANSACRegressor
from sklearn.model_selection import train_test_split

# Example with robust regression models
X = df[numeric_cols]
y = df['target']  # Replace with your target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Huber Regression (robust to outliers)
huber = HuberRegressor(epsilon=1.35)
huber.fit(X_train, y_train)
print(f"Huber Regression score: {huber.score(X_test, y_test):.4f}")

# RANSAC Regression (extremely robust to outliers)
ransac = RANSACRegressor(random_state=42)
ransac.fit(X_train, y_train)
print(f"RANSAC Regression score: {ransac.score(X_test, y_test):.4f}")

# Random Forest (naturally robust to outliers)
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
print(f"Random Forest score: {rf.score(X_test, y_test):.4f}")
```

### 5. Create Separate Models

Treat outliers as a separate group:

```python
# Identify outliers
outlier_mask = (np.abs(stats.zscore(df[numeric_cols])) > 3).any(axis=1)

# Create separate datasets
df_inliers = df[~outlier_mask]
df_outliers = df[outlier_mask]

print(f"Inliers: {len(df_inliers)}, Outliers: {len(df_outliers)}")

# You can now train separate models for each group
```

## Handling Strategy Comparison

| Strategy | Pros | Cons | Best Used When |
|----------|------|------|---------------|
| Removal | Simple, effective | Data loss | Outliers are errors |
| Capping | Preserves data points | Distorts distribution | Many extreme values exist |
| Transformation | Preserves relationships | Changes interpretation | Data is highly skewed |
| Robust Models | No data modification | May be less accurate | Outliers are valid but extreme |
| Separate Models | Handles different regimes | Complex implementation | Outliers represent different population |

## Best Practices

1. **Investigate outliers** before removing them
2. **Document decisions** about outlier treatment
3. **Consider domain knowledge** when defining what constitutes an outlier
4. **Use visualization** to understand the nature of outliers
5. **Compare multiple detection methods**
6. **Test sensitivity** of model to different outlier strategies
7. **Preserve original data** while working with cleaned versions 