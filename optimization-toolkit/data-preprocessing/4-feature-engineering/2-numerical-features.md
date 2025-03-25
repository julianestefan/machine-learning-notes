# Numerical Feature Engineering

Numerical feature engineering involves transforming and creating features from continuous or discrete numeric data to improve model performance.

## Data Scaling and Normalization

Many machine learning algorithms perform better when features are on similar scales.

### Standardization (Z-score)

Transforms features to have mean=0 and standard deviation=1:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Select numeric columns for scaling
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

# Initialize the scaler
scaler = StandardScaler()

# Fit and transform the data
df_scaled = pd.DataFrame(
    scaler.fit_transform(df[numeric_cols]),
    columns=numeric_cols,
    index=df.index
)

# Compare original and scaled distributions
feature = numeric_cols[0]  # Choose first numeric feature
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(df[feature], kde=True, ax=ax1)
ax1.set_title(f'Original {feature} Distribution')
sns.histplot(df_scaled[feature], kde=True, ax=ax2)
ax2.set_title(f'Standardized {feature} Distribution')
plt.tight_layout()
plt.show()

print(f"Original {feature} - Mean: {df[feature].mean():.2f}, Std: {df[feature].std():.2f}")
print(f"Scaled {feature} - Mean: {df_scaled[feature].mean():.2f}, Std: {df_scaled[feature].std():.2f}")
```

### Min-Max Scaling

Scales features to a fixed range, usually [0,1]:

```python
from sklearn.preprocessing import MinMaxScaler

# Initialize the scaler
min_max_scaler = MinMaxScaler()

# Fit and transform
df_minmax = pd.DataFrame(
    min_max_scaler.fit_transform(df[numeric_cols]),
    columns=numeric_cols,
    index=df.index
)

# Compare distributions
feature = numeric_cols[0]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(df[feature], kde=True, ax=ax1)
ax1.set_title(f'Original {feature} Distribution')
sns.histplot(df_minmax[feature], kde=True, ax=ax2)
ax2.set_title(f'Min-Max Scaled {feature} Distribution')
plt.tight_layout()
plt.show()

print(f"Min-Max {feature} - Min: {df_minmax[feature].min():.2f}, Max: {df_minmax[feature].max():.2f}")
```

### Robust Scaling

Scales using median and interquartile range; less affected by outliers:

```python
from sklearn.preprocessing import RobustScaler

# Initialize the scaler
robust_scaler = RobustScaler()

# Fit and transform
df_robust = pd.DataFrame(
    robust_scaler.fit_transform(df[numeric_cols]),
    columns=numeric_cols,
    index=df.index
)

# Compare with outliers
feature_with_outliers = numeric_cols[0]  # Choose feature with outliers

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
sns.boxplot(y=df[feature_with_outliers], ax=ax1)
ax1.set_title('Original')
sns.boxplot(y=df_scaled[feature_with_outliers], ax=ax2)
ax2.set_title('StandardScaler')
sns.boxplot(y=df_robust[feature_with_outliers], ax=ax3)
ax3.set_title('RobustScaler')
plt.tight_layout()
plt.show()
```

### Scaling Comparison

| Method | Formula | Strengths | Weaknesses | Best For |
|--------|---------|-----------|------------|----------|
| Z-score | (x - μ) / σ | Preserves outlier relationships | Sensitive to outliers | Normal-like distributions |
| Min-Max | (x - min) / (max - min) | Intuitive, preserves relationships | Very sensitive to outliers | Bounded features without outliers |
| Robust | (x - median) / IQR | Resistant to outliers | May lose some information | Data with outliers |

## Distribution Transformations

Many models perform better when features follow normal-like distributions.

### Logarithmic Transformation

Useful for right-skewed data:

```python
import numpy as np

# Add small constant to avoid log(0)
epsilon = 1e-8

# Apply log transformation
df['log_income'] = np.log(df['income'] + epsilon)

# Compare distributions
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(df['income'], kde=True, ax=ax1)
ax1.set_title('Original Income Distribution')
sns.histplot(df['log_income'], kde=True, ax=ax2)
ax2.set_title('Log-transformed Income Distribution')
plt.tight_layout()
plt.show()

# Check skewness before and after
print(f"Original Skewness: {df['income'].skew():.4f}")
print(f"Log-transformed Skewness: {df['log_income'].skew():.4f}")
```

### Square Root Transformation

For moderately skewed data:

```python
# Apply square root transformation
df['sqrt_distance'] = np.sqrt(df['distance'])

# Compare distributions
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(df['distance'], kde=True, ax=ax1)
ax1.set_title('Original Distance Distribution')
sns.histplot(df['sqrt_distance'], kde=True, ax=ax2)
ax2.set_title('Square Root-transformed Distance Distribution')
plt.tight_layout()
plt.show()
```

### Box-Cox Transformation

Applies the optimal power transformation:

```python
from scipy import stats

# Ensure data is positive
feature = 'positive_feature'
if (df[feature] <= 0).any():
    df[f'{feature}_adj'] = df[feature] - df[feature].min() + 1
else:
    df[f'{feature}_adj'] = df[feature]

# Apply Box-Cox
transformed_data, lambda_value = stats.boxcox(df[f'{feature}_adj'])
df[f'{feature}_boxcox'] = transformed_data

print(f"Optimal lambda value: {lambda_value:.4f}")
# Lambda near 0 -> log transform, Lambda = 1 -> no transformation

# Compare distributions
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(df[f'{feature}_adj'], kde=True, ax=ax1)
ax1.set_title(f'Original {feature} Distribution')
sns.histplot(df[f'{feature}_boxcox'], kde=True, ax=ax2)
ax2.set_title(f'Box-Cox Transformed Distribution (λ={lambda_value:.4f})')
plt.tight_layout()
plt.show()
```

### Yeo-Johnson Transformation

Similar to Box-Cox but works with negative values:

```python
from sklearn.preprocessing import PowerTransformer

# Initialize transformer
pt = PowerTransformer(method='yeo-johnson')

# Apply transformation
feature_df = df[[feature]].copy()
feature_df[f'{feature}_yeojohnson'] = pt.fit_transform(feature_df[[feature]])

# Compare distributions
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(feature_df[feature], kde=True, ax=ax1)
ax1.set_title(f'Original {feature} Distribution')
sns.histplot(feature_df[f'{feature}_yeojohnson'], kde=True, ax=ax2)
ax2.set_title('Yeo-Johnson Transformed Distribution')
plt.tight_layout()
plt.show()
```

## Creating Binary Features

Converting numeric data to binary indicators:

```python
# Simple threshold
df['is_tall'] = (df['height'] > 180).astype(int)

# Multiple thresholds for complex conditions
df['temp_category'] = 0  # Default value
df.loc[df['temperature'] < 0, 'temp_category'] = 1  # Freezing
df.loc[(df['temperature'] >= 0) & (df['temperature'] < 15), 'temp_category'] = 2  # Cold
df.loc[(df['temperature'] >= 15) & (df['temperature'] < 25), 'temp_category'] = 3  # Moderate
df.loc[df['temperature'] >= 25, 'temp_category'] = 4  # Hot

# Convert continuous variable to binary based on its relationship with target
avg_target_by_value = df.groupby('numeric_feature')['target'].mean()
threshold = avg_target_by_value.median()
df['high_target_probability'] = (df['numeric_feature'] > threshold).astype(int)
```

## Creating Binned Features

Group continuous data into discrete bins:

```python
# Equal-width binning
df['age_bin_equal_width'] = pd.cut(
    df['age'], 
    bins=5,
    labels=['Very Young', 'Young', 'Middle', 'Senior', 'Elderly']
)

# Equal-frequency binning (quantiles)
df['income_quantile'] = pd.qcut(
    df['income'],
    q=4,
    labels=['Low', 'Medium', 'High', 'Very High']
)

# Custom binning
custom_bins = [-np.inf, 0, 18, 35, 50, 65, np.inf]
custom_labels = ['Invalid', 'Child', 'Young Adult', 'Adult', 'Middle Aged', 'Senior']
df['age_group'] = pd.cut(df['age'], bins=custom_bins, labels=custom_labels)

# Visualize bin distributions
plt.figure(figsize=(10, 6))
sns.countplot(y='age_group', data=df, order=df['age_group'].value_counts().index)
plt.title('Distribution of Age Groups')
plt.tight_layout()
plt.show()
```

## Creating Interaction Features

Sometimes relationships between features matter:

```python
# Multiplication
df['area'] = df['length'] * df['width']
df['volume'] = df['length'] * df['width'] * df['height']
df['bmi'] = df['weight'] / (df['height'] / 100) ** 2

# Ratios
df['efficiency'] = df['output'] / df['input']
df['profit_margin'] = df['profit'] / df['revenue']

# Differences
df['age_gap'] = df['father_age'] - df['child_age']
df['price_difference'] = df['selling_price'] - df['purchase_price']

# Polynomial features
df['age_squared'] = df['age'] ** 2
df['income_cubed'] = df['income'] ** 3

# Using scikit-learn for systematic generation
from sklearn.preprocessing import PolynomialFeatures

# Create interaction terms up to degree 2
features = ['feature_1', 'feature_2', 'feature_3']
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(df[features])

# Convert to DataFrame with appropriate column names
feature_names = poly.get_feature_names_out(features)
poly_df = pd.DataFrame(poly_features, columns=feature_names, index=df.index)

# Join with original DataFrame
df = pd.concat([df, poly_df], axis=1)
```

## Aggregating Features

Creating summary features from multiple columns:

```python
# Statistical aggregations
df['mean_score'] = df[['score_1', 'score_2', 'score_3']].mean(axis=1)
df['max_score'] = df[['score_1', 'score_2', 'score_3']].max(axis=1)
df['min_score'] = df[['score_1', 'score_2', 'score_3']].min(axis=1)
df['range_score'] = df['max_score'] - df['min_score']
df['sum_score'] = df[['score_1', 'score_2', 'score_3']].sum(axis=1)
df['median_score'] = df[['score_1', 'score_2', 'score_3']].median(axis=1)
df['std_score'] = df[['score_1', 'score_2', 'score_3']].std(axis=1)

# Group-based aggregations
grouped_stats = df.groupby('category').agg({
    'numeric_feature': ['mean', 'median', 'std', 'min', 'max']
})

# Merge aggregated stats back to original dataframe
for stat in ['mean', 'median', 'std', 'min', 'max']:
    df[f'category_{stat}'] = df['category'].map(
        grouped_stats[('numeric_feature', stat)]
    )

# Calculate difference from group mean
df['diff_from_category_mean'] = df['numeric_feature'] - df['category_mean']
```

## Handling Special Values

Special numeric values often need custom treatment:

```python
# Handling zeros
df['log_zeros_handled'] = np.log1p(df['feature_with_zeros'])  # log(1+x)

# Handling negative values
df['sqrt_neg_handled'] = np.sqrt(np.abs(df['feature_with_negatives']))
df['neg_indicator'] = (df['feature_with_negatives'] < 0).astype(int)

# Handling infinities
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df['feature_with_infs'].fillna(df['feature_with_infs'].median(), inplace=True)

# Capping extreme values (winsorization)
lower_percentile = df['feature_with_outliers'].quantile(0.01)
upper_percentile = df['feature_with_outliers'].quantile(0.99)
df['feature_winsorized'] = df['feature_with_outliers'].clip(lower_percentile, upper_percentile)
```

## Feature Crossing

Combining multiple numeric features:

```python
# Binned feature crossing
df['height_binned'] = pd.qcut(df['height'], q=3, labels=['Short', 'Medium', 'Tall'])
df['weight_binned'] = pd.qcut(df['weight'], q=3, labels=['Light', 'Medium', 'Heavy'])
df['height_weight_cross'] = df['height_binned'].astype(str) + '_' + df['weight_binned'].astype(str)

# Convert crossing to one-hot encoding if needed
crossed_dummies = pd.get_dummies(df['height_weight_cross'], prefix='hw_cross')
df = pd.concat([df, crossed_dummies], axis=1)
```

## Numerical Feature Selection

Not all numerical features are equally useful:

```python
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression

# Prepare data
X = df[numeric_cols]
y = df['target']

# Select features based on F-statistic
f_selector = SelectKBest(f_regression, k=5)  # Select top 5 features
X_f_selected = f_selector.fit_transform(X, y)

# Get selected feature names
f_support = f_selector.get_support()
f_feature_names = X.columns[f_support]
print("Features selected using F-statistic:", f_feature_names.tolist())

# Select features based on mutual information
mi_selector = SelectKBest(mutual_info_regression, k=5)
X_mi_selected = mi_selector.fit_transform(X, y)

# Get selected feature names
mi_support = mi_selector.get_support()
mi_feature_names = X.columns[mi_support]
print("Features selected using mutual information:", mi_feature_names.tolist())

# Plot feature importance
plt.figure(figsize=(12, 6))
plt.barh(X.columns, f_selector.scores_)
plt.title('Feature Importance (F-statistic)')
plt.xlabel('F-Score')
plt.tight_layout()
plt.show()
```

## Evaluating Numerical Features

Always test whether new features improve model performance:

```python
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression

# Define function to evaluate features
def evaluate_features(X, y, cv=5):
    """Evaluate feature set using cross-validation"""
    model = LinearRegression()
    scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
    return scores.mean(), scores.std()

# Evaluate original features
original_score, original_std = evaluate_features(X, y)
print(f"Original features: R² = {original_score:.3f} (±{original_std:.3f})")

# Evaluate engineered features
X_engineered = pd.concat([X, df['log_income'], df['area'], df['bmi']], axis=1)
engineered_score, engineered_std = evaluate_features(X_engineered, y)
print(f"Engineered features: R² = {engineered_score:.3f} (±{engineered_std:.3f})")

# Improvement
improvement = (engineered_score - original_score) / original_score * 100
print(f"Improvement: {improvement:.1f}%")
```

## Numerical Feature Engineering Best Practices

1. **Always check distributions** before applying transformations
2. **Visualize before and after** to validate transformations
3. **Handle special cases** (zeros, negatives, infinities) appropriately
4. **Create multiple variants** of important features
5. **Test impact** of each transformation on model performance
6. **Scale after splitting** training and test data to prevent leakage
7. **Document transformation logic** for reproducibility in production
8. **Consider domain knowledge** when creating interaction features
9. **Avoid creating too many** numerical features to prevent overfitting 