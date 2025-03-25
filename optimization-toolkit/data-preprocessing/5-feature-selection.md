# Feature Selection Techniques

Feature selection is the process of identifying and selecting the most relevant features for a machine learning model. Proper feature selection can improve model performance, reduce overfitting, decrease training time, and enhance interpretability.

## Why Feature Selection is Important

- **Improved Model Performance**: Removing irrelevant and redundant features can improve model accuracy
- **Reduced Overfitting**: Fewer features often lead to models that generalize better
- **Faster Training**: Lower dimensionality reduces computational costs
- **Better Interpretability**: Simpler models with fewer features are easier to understand
- **Reduced Data Collection Costs**: Identifying only necessary features can reduce future data acquisition expenses

## Types of Feature Selection Methods

There are three main categories of feature selection methods:

1. **Filter Methods**: Select features based on statistical measures, independent of the model
2. **Wrapper Methods**: Use a specific machine learning algorithm to evaluate feature subsets
3. **Embedded Methods**: Feature selection occurs as part of the model training process

Let's explore each method in detail with Python code examples.

## Filter Methods

Filter methods use statistical measures to score the correlation or dependence between input features and the target variable. They are generally faster and less computationally expensive than wrapper methods.

### Variance Threshold

Removes features with low variance, assuming that features with little variance across samples might not be informative.

```python
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.datasets import load_breast_cancer

# Load example dataset
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names

print(f"Original shape: {X.shape}")

# Create a variance threshold selector
selector = VarianceThreshold(threshold=0.2)  # Features with a variance below 0.2 will be removed
X_selected = selector.fit_transform(X)

# Get selected feature names
selected_indices = selector.get_support(indices=True)
selected_features = [feature_names[i] for i in selected_indices]

print(f"Transformed shape: {X_selected.shape}")
print(f"Selected features: {selected_features}")
```

### Correlation Matrix

Identifies and removes highly correlated features to reduce redundancy.

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Convert to pandas DataFrame for better handling
X_df = pd.DataFrame(X, columns=feature_names)

# Calculate correlation matrix
corr_matrix = X_df.corr()

# Plot correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, linewidths=0.5)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.show()

# Function to remove highly correlated features
def remove_correlated_features(df, threshold=0.8):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    df_reduced = df.drop(columns=to_drop)
    return df_reduced, to_drop

# Remove highly correlated features
X_df_reduced, dropped_features = remove_correlated_features(X_df, threshold=0.9)
print(f"Original features: {X_df.shape[1]}")
print(f"Features after correlation removal: {X_df_reduced.shape[1]}")
print(f"Dropped features: {dropped_features}")
```

### Statistical Tests

Selects features based on statistical significance of their relationship with the target variable.

```python
from sklearn.feature_selection import SelectKBest, f_classif, chi2, mutual_info_classif

# Different statistical tests for feature selection
# 1. ANOVA F-test for classification
selector_f = SelectKBest(f_classif, k=10)
X_kbest_f = selector_f.fit_transform(X, y)

# 2. Chi-squared test (requires non-negative features)
X_pos = np.abs(X)  # Making features non-negative for chi2
selector_chi2 = SelectKBest(chi2, k=10)
X_kbest_chi2 = selector_chi2.fit_transform(X_pos, y)

# 3. Mutual Information (measure of dependency between variables)
selector_mi = SelectKBest(mutual_info_classif, k=10)
X_kbest_mi = selector_mi.fit_transform(X, y)

# Get selected feature names and scores
def get_selected_features(selector, feature_names):
    selected_indices = selector.get_support(indices=True)
    selected_features = [feature_names[i] for i in selected_indices]
    scores = selector.scores_
    return selected_features, scores

f_selected, f_scores = get_selected_features(selector_f, feature_names)
chi2_selected, chi2_scores = get_selected_features(selector_chi2, feature_names)
mi_selected, mi_scores = get_selected_features(selector_mi, feature_names)

# Print top 10 features by F-value
print("Top 10 features by F-test:")
for feature, score in sorted(zip(feature_names, f_scores), key=lambda x: x[1], reverse=True)[:10]:
    print(f"{feature}: {score:.3f}")

# Plot feature importance based on F-test
plt.figure(figsize=(12, 6))
plt.barh(range(len(f_scores)), f_scores, align='center')
plt.yticks(range(len(f_scores)), feature_names)
plt.xlabel('F-score')
plt.title('Feature Importance by F-test')
plt.tight_layout()
plt.show()
```

### Filter Methods for Regression

For regression problems, we can use different measures.

```python
from sklearn.feature_selection import f_regression, mutual_info_regression
import sklearn.datasets as datasets

# Load regression dataset
boston = datasets.fetch_california_housing()
X_reg = boston.data
y_reg = boston.target
reg_feature_names = boston.feature_names

# F-test for regression
selector_f_reg = SelectKBest(f_regression, k=5)
X_kbest_f_reg = selector_f_reg.fit_transform(X_reg, y_reg)

# Mutual Information for regression
selector_mi_reg = SelectKBest(mutual_info_regression, k=5)
X_kbest_mi_reg = selector_mi_reg.fit_transform(X_reg, y_reg)

# Get selected features and scores
f_reg_selected, f_reg_scores = get_selected_features(selector_f_reg, reg_feature_names)
mi_reg_selected, mi_reg_scores = get_selected_features(selector_mi_reg, reg_feature_names)

print("Top features by F-test (regression):")
for feature, score in sorted(zip(reg_feature_names, f_reg_scores), key=lambda x: x[1], reverse=True):
    print(f"{feature}: {score:.3f}")
```

## Wrapper Methods

Wrapper methods evaluate subsets of features by training a model on each subset. They often provide better performance but are computationally expensive.

### Recursive Feature Elimination (RFE)

Recursively eliminates the least important features.

```python
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

# Create a base model
base_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Create RFE model and select attributes
rfe = RFE(estimator=base_model, n_features_to_select=10, step=1)
rfe.fit(X, y)

# Get selected features
selected_features_rfe = [feature for feature, selected in zip(feature_names, rfe.support_) if selected]
print("Features selected by RFE:")
for i, feature in enumerate(selected_features_rfe):
    print(f"{i+1}. {feature} (Rank: {rfe.ranking_[list(feature_names).index(feature)]})")

# Plotting feature ranking
plt.figure(figsize=(12, 6))
plt.barh(range(len(rfe.ranking_)), rfe.ranking_, align='center')
plt.yticks(range(len(rfe.ranking_)), feature_names)
plt.xlabel('Ranking (lower is better)')
plt.title('Feature Ranking by RFE')
plt.tight_layout()
plt.show()
```

### Sequential Feature Selection

Adds or removes features one at a time based on model performance.

```python
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier

# Create a base model
knn = KNeighborsClassifier(n_neighbors=3)

# Forward Sequential Feature Selection
sfs_forward = SequentialFeatureSelector(knn, n_features_to_select=10, direction='forward', cv=5)
sfs_forward.fit(X, y)

# Backward Sequential Feature Selection
sfs_backward = SequentialFeatureSelector(knn, n_features_to_select=10, direction='backward', cv=5)
sfs_backward.fit(X, y)

# Get selected features
forward_selected = [feature for feature, selected in zip(feature_names, sfs_forward.support_) if selected]
backward_selected = [feature for feature, selected in zip(feature_names, sfs_backward.support_) if selected]

print("Features selected by Forward SFS:")
print(forward_selected)
print("\nFeatures selected by Backward SFS:")
print(backward_selected)

# Compare shared features
shared_features = set(forward_selected).intersection(set(backward_selected))
print(f"\nFeatures selected by both methods: {shared_features}")
```

### Cross-Validation Feature Selection

Evaluates feature subsets using cross-validation to choose the best subset.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from itertools import combinations

def feature_selection_cv(X, y, feature_names, max_features=10, cv=5):
    best_score = 0.0
    best_features = []
    model = LogisticRegression(max_iter=1000, solver='liblinear')
    
    # Start with top 3 features from f_classif
    selector = SelectKBest(f_classif, k=3)
    selector.fit(X, y)
    initial_features = [feature_names[i] for i in selector.get_support(indices=True)]
    current_features = initial_features.copy()
    
    # Forward selection with cross-validation
    remaining_features = [f for f in feature_names if f not in current_features]
    
    print(f"Starting with features: {current_features}")
    print(f"Initial CV score: {cross_val_score(model, X[:, [list(feature_names).index(f) for f in current_features]], y, cv=cv).mean():.4f}")
    
    while len(current_features) < max_features:
        best_new_score = 0.0
        best_new_feature = None
        
        # Try each remaining feature
        for feature in remaining_features:
            candidate_features = current_features + [feature]
            feature_indices = [list(feature_names).index(f) for f in candidate_features]
            score = cross_val_score(model, X[:, feature_indices], y, cv=cv).mean()
            
            if score > best_new_score:
                best_new_score = score
                best_new_feature = feature
        
        # If no improvement, break
        if best_new_score <= best_score and len(current_features) > 0:
            break
            
        # Add the best new feature
        if best_new_feature:
            current_features.append(best_new_feature)
            remaining_features.remove(best_new_feature)
            best_score = best_new_score
            print(f"Added feature: {best_new_feature}, CV score: {best_score:.4f}")
    
    return current_features, best_score

# Run cross-validation feature selection
selected_cv_features, cv_score = feature_selection_cv(X, y, feature_names, max_features=10)
print(f"\nFinal selected features: {selected_cv_features}")
print(f"Final CV score: {cv_score:.4f}")
```

## Embedded Methods

Embedded methods perform feature selection as part of the model training process, combining the advantages of filter and wrapper methods.

### LASSO (L1 Regularization)

Adds a penalty term to the loss function that encourages sparsity in feature coefficients.

```python
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Find optimal alpha using cross-validation
lasso_cv = LassoCV(cv=5, random_state=42, max_iter=10000)
lasso_cv.fit(X_scaled, y)
optimal_alpha = lasso_cv.alpha_

print(f"Optimal alpha: {optimal_alpha:.6f}")

# Fit LASSO with optimal alpha
lasso = Lasso(alpha=optimal_alpha, random_state=42, max_iter=10000)
lasso.fit(X_scaled, y)

# Get feature importance
feature_importance = np.abs(lasso.coef_)
feature_importance_normalized = feature_importance / feature_importance.max()

# Display features with non-zero coefficients
selected_features_lasso = [(feature, coef) for feature, coef in zip(feature_names, lasso.coef_) if coef != 0]
print(f"LASSO selected {len(selected_features_lasso)} features out of {len(feature_names)}")

for feature, coef in sorted(selected_features_lasso, key=lambda x: abs(x[1]), reverse=True):
    print(f"{feature}: {coef:.6f}")

# Plot LASSO coefficients
plt.figure(figsize=(12, 6))
plt.bar(range(len(feature_importance_normalized)), feature_importance_normalized)
plt.xticks(range(len(feature_importance_normalized)), feature_names, rotation=90)
plt.xlabel('Feature')
plt.ylabel('Normalized Coefficient')
plt.title('Feature Importance from LASSO')
plt.tight_layout()
plt.show()
```

### Tree-Based Feature Importance

Uses feature importance scores from tree-based models like Random Forest and Gradient Boosting.

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import pandas as pd

# Random Forest Feature Importance
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)
rf_importances = rf.feature_importances_

# Gradient Boosting Feature Importance
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb.fit(X, y)
gb_importances = gb.feature_importances_

# Create a DataFrame for comparison
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Random Forest': rf_importances,
    'Gradient Boosting': gb_importances
})

# Sort by Random Forest importance
importance_df = importance_df.sort_values('Random Forest', ascending=False)

print("Feature importance comparison:")
print(importance_df)

# Plot feature importance comparison
importance_df_melted = pd.melt(importance_df, id_vars=['Feature'], 
                               value_vars=['Random Forest', 'Gradient Boosting'],
                               var_name='Model', value_name='Importance')

plt.figure(figsize=(14, 8))
sns.barplot(x='Feature', y='Importance', hue='Model', data=importance_df_melted)
plt.xticks(rotation=90)
plt.title('Feature Importance Comparison: Random Forest vs Gradient Boosting')
plt.tight_layout()
plt.show()

# Select top N features based on Random Forest importance
def select_top_features(feature_names, importances, top_n=10):
    # Get indices of sorted importances
    sorted_indices = np.argsort(importances)[::-1]
    # Get feature names in order of importance
    top_features = [feature_names[i] for i in sorted_indices[:top_n]]
    return top_features

top_rf_features = select_top_features(feature_names, rf_importances, top_n=10)
print("\nTop 10 features selected by Random Forest:")
print(top_rf_features)
```

### Permutation Feature Importance

Measures importance by observing how model performance changes when feature values are randomly shuffled.

```python
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Calculate permutation importance
result = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)
perm_importance = result.importances_mean

# Create a DataFrame for visualization
perm_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': perm_importance
}).sort_values('Importance', ascending=False)

print("Permutation Feature Importance:")
print(perm_importance_df)

# Plot permutation importance
plt.figure(figsize=(12, 6))
sns.barplot(x='Importance', y='Feature', data=perm_importance_df.head(15))
plt.title('Permutation Feature Importance')
plt.tight_layout()
plt.show()
```

## Feature Selection with Scikit-learn Pipeline

Integrating feature selection into a Machine Learning pipeline:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Create a pipeline with preprocessing, feature selection, and model
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42))),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Define parameters for grid search
param_grid = {
    'feature_selection__threshold': ['mean', '0.5*mean', '2*mean'],
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 10, 20]
}

# Perform grid search
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X, y)

# Print best parameters
print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)

# Get selected features from best model
best_pipeline = grid_search.best_estimator_
feature_selector = best_pipeline.named_steps['feature_selection']
selected_mask = feature_selector.get_support()
selected_features_pipeline = [feature for feature, selected in zip(feature_names, selected_mask) if selected]

print("Features selected by the best pipeline:")
print(selected_features_pipeline)
```

## Evaluating Feature Selection

Comparing model performance before and after feature selection:

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Define classifiers to evaluate
classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
}

# Feature selection methods
# 1. All original features
X_all = X

# 2. Feature selection with RFE (10 features)
rfe = RFE(estimator=RandomForestClassifier(n_estimators=100, random_state=42), n_features_to_select=10)
X_rfe = rfe.fit_transform(X, y)

# 3. Feature selection with LASSO
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
lasso = Lasso(alpha=0.01, random_state=42)
lasso.fit(X_scaled, y)
selected_mask = lasso.coef_ != 0
X_lasso = X[:, selected_mask]

# 4. Feature selection with tree-based importance
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)
selector = SelectFromModel(rf, threshold='mean')
X_rf = selector.fit_transform(X, y)

# Evaluate each classifier with each feature set
results = []

for cls_name, classifier in classifiers.items():
    # All features
    all_score = cross_val_score(classifier, X_all, y, cv=5, scoring='accuracy').mean()
    
    # RFE features
    rfe_score = cross_val_score(classifier, X_rfe, y, cv=5, scoring='accuracy').mean()
    
    # LASSO features
    lasso_score = cross_val_score(classifier, X_lasso, y, cv=5, scoring='accuracy').mean()
    
    # Random Forest features
    rf_score = cross_val_score(classifier, X_rf, y, cv=5, scoring='accuracy').mean()
    
    results.append({
        'Classifier': cls_name,
        'All Features': all_score,
        'RFE Features': rfe_score,
        'LASSO Features': lasso_score,
        'RF Features': rf_score
    })

# Display results as a DataFrame
results_df = pd.DataFrame(results)
print(results_df)

# Plot results
results_melted = pd.melt(results_df, id_vars=['Classifier'], 
                          value_vars=['All Features', 'RFE Features', 'LASSO Features', 'RF Features'],
                          var_name='Feature Selection', value_name='Accuracy')

plt.figure(figsize=(14, 8))
sns.barplot(x='Classifier', y='Accuracy', hue='Feature Selection', data=results_melted)
plt.title('Classifier Performance with Different Feature Selection Methods')
plt.ylim(0.8, 1.0)  # Adjust as needed
plt.tight_layout()
plt.show()
```

## Feature Selection for Dimensionality Reduction

When dealing with high-dimensional data:

```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Apply PCA for dimensionality reduction
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Apply t-SNE for dimensionality reduction
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Plot PCA and t-SNE results
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot PCA
scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.8)
axes[0].set_title('PCA Dimensionality Reduction')
axes[0].set_xlabel('Principal Component 1')
axes[0].set_ylabel('Principal Component 2')
legend1 = axes[0].legend(*scatter1.legend_elements(), title="Classes")
axes[0].add_artist(legend1)

# Plot t-SNE
scatter2 = axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.8)
axes[1].set_title('t-SNE Dimensionality Reduction')
axes[1].set_xlabel('t-SNE Dimension 1')
axes[1].set_ylabel('t-SNE Dimension 2')
legend2 = axes[1].legend(*scatter2.legend_elements(), title="Classes")
axes[1].add_artist(legend2)

plt.tight_layout()
plt.show()

# Compare explained variance ratio in PCA
pca_full = PCA().fit(X)
plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(pca_full.explained_variance_ratio_))
plt.axhline(y=0.95, color='r', linestyle='-')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by PCA Components')
plt.grid(True)
plt.show()
```

## Best Practices for Feature Selection

1. **Understand your data**: Explore and visualize your data before applying feature selection.

2. **Use domain knowledge**: Incorporate domain expertise when selecting features.

3. **Try multiple methods**: Different feature selection methods may yield different results.

4. **Cross-validate**: Always validate your feature selection process with cross-validation.

5. **Consider stability**: Check if feature selection is stable across different data samples.

6. **Watch for multicollinearity**: Be aware of correlations between features.

7. **Balance complexity and performance**: More features aren't always better; aim for the simplest model that achieves good performance.

8. **Validate with independent test set**: Ensure selected features generalize well to unseen data.

9. **Automate with pipelines**: Use scikit-learn pipelines to integrate feature selection into your ML workflow.

10. **Document your process**: Keep track of feature selection decisions for reproducibility. 