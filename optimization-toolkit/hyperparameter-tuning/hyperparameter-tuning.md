# Hyperparameter Tuning: A Comprehensive Guide

Hyperparameter tuning is the process of finding the optimal configuration of hyperparameters for a machine learning algorithm. Unlike model parameters that are learned during training, hyperparameters are set before the learning process begins and significantly influence model performance.

## Table of Contents

1. [Introduction to Hyperparameter Tuning](#introduction-to-hyperparameter-tuning)
2. [Grid Search](#grid-search)
3. [Random Search](#random-search)
4. [Informed Search Methods](#informed-search-methods)
   - [Coarse to Fine Tuning](#coarse-to-fine-tuning)
   - [Bayesian Optimization](#bayesian-optimization)
   - [Genetic Algorithms](#genetic-algorithms)
5. [Automated ML Frameworks](#automated-ml-frameworks)
   - [TPOT](#tpot)
6. [Hyperparameter Tuning Best Practices](#hyperparameter-tuning-best-practices)
7. [Hyperparameters by Algorithm](#hyperparameters-by-algorithm)

## Introduction to Hyperparameter Tuning

Hyperparameter tuning is crucial for maximizing model performance. Poor hyperparameter choices can result in underfitting or overfitting, regardless of the quality of your data or model architecture.

### Key Concepts

- **Hyperparameters vs. Parameters**: Hyperparameters are set before training (e.g., learning rate, tree depth), while parameters are learned during training (e.g., weights in neural networks)
- **Search Space**: The range of possible values for each hyperparameter
- **Objective Function**: The metric used to evaluate model performance (e.g., accuracy, RMSE)
- **Validation Strategy**: Typically k-fold cross-validation to ensure robust performance estimates

### General Approaches

- **Manual Tuning**: Using domain knowledge and experimentation to set hyperparameters
- **Automated Tuning**: Systematic exploration of the hyperparameter space

### Best Practices Before Tuning

1. **Read documentation** to understand hyperparameter interactions and constraints
2. **Avoid extreme values** (e.g., very low number of trees in Random Forest, 1 neighbor in KNN)
3. **Start with defaults** and make incremental changes
4. **Consider computational resources** available for tuning

## Grid Search

Grid search is the most straightforward hyperparameter tuning method, involving an exhaustive search over specified parameter values.

### How It Works

1. Define a grid of hyperparameter values to explore
2. Train and evaluate a model for each combination of hyperparameters
3. Select the combination with the best performance

### Advantages and Disadvantages

#### Advantages
- Simple to implement and understand
- Guaranteed to find the best combination within the defined grid
- Easily parallelizable

#### Disadvantages
- Computationally expensive for large hyperparameter spaces
- Suffers from the "curse of dimensionality" as the number of hyperparameters increases
- Uninformed: one model doesn't help in creating the next one

### Implementation with Scikit-learn

```python
# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt

# Assume X and y are already defined
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest Classifier with specified criterion
rf_classifier = RandomForestClassifier(criterion='entropy', random_state=42)

# Define parameter grid
param_grid = {
    "max_depth": [2, 4, 8, 15], 
    "max_features": ['auto', 'sqrt'],
    "n_estimators": [100, 200, 300]
}

# Create a GridSearchCV object
grid_search = GridSearchCV(
    estimator=rf_classifier,
    param_grid=param_grid,
    scoring='roc_auc',
    n_jobs=-1,  # Use all available cores
    cv=5,       # 5-fold cross-validation
    verbose=1,
    refit=True, 
    return_train_score=True
)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Print best parameters and score
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score: {:.3f}".format(grid_search.best_score_))

# Get the best model
best_model = grid_search.best_estimator_

# Evaluate on test set
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# Print test set performance
print("Test set accuracy: {:.3f}".format(accuracy_score(y_test, y_pred)))
print("Test set ROC AUC: {:.3f}".format(roc_auc_score(y_test, y_pred_proba)))
```

### Analyzing Grid Search Results

```python
# Convert the cv_results_ to a DataFrame for easier analysis
cv_results_df = pd.DataFrame(grid_search.cv_results_)

# View the top-performing parameter combinations
best_results = cv_results_df.sort_values(by='rank_test_score').head(5)
print(best_results[['params', 'mean_test_score', 'std_test_score']])

# Extract the best row by index
best_row = cv_results_df.loc[[grid_search.best_index_]]
print(best_row)

# Visualize the impact of max_depth on performance
plt.figure(figsize=(10, 6))
for features in param_grid['max_features']:
    for n_est in param_grid['n_estimators']:
        # Filter results for this combination
        mask = (cv_results_df['param_max_features'] == features) & \
               (cv_results_df['param_n_estimators'] == n_est)
        
        # Plot the line
        plt.plot(
            cv_results_df.loc[mask, 'param_max_depth'],
            cv_results_df.loc[mask, 'mean_test_score'],
            marker='o',
            label=f"max_features={features}, n_estimators={n_est}"
        )

plt.title('Effect of max_depth on Model Performance')
plt.xlabel('max_depth')
plt.ylabel('Mean Test Score (ROC AUC)')
plt.legend()
plt.grid(True)
plt.show()
```

### Grid Search for Regression Models

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Create Random Forest Regressor
rf_regressor = RandomForestRegressor(random_state=42)

# Define parameter grid for regressor
param_grid_rf = {
    'n_estimators': [100, 350, 500],
    'max_features': ['log2', 'auto', 'sqrt'],
    'min_samples_leaf': [2, 10, 30]
}

# Create GridSearchCV for regression
grid_rf = GridSearchCV(
    estimator=rf_regressor,
    param_grid=param_grid_rf,
    scoring='neg_mean_squared_error',  # Note the 'neg_' prefix for minimization
    cv=5,
    verbose=1,
    n_jobs=-1
)

# Fit to training data
grid_rf.fit(X_train, y_train)

# Get best estimator
best_rf_model = grid_rf.best_estimator_

# Predict on test set
y_pred = best_rf_model.predict(X_test)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print('Test RMSE of best model: {:.3f}'.format(rmse))
```

## Random Search

Random search samples random combinations of hyperparameters rather than exhaustively trying all combinations. This is often more efficient than grid search, especially when not all hyperparameters are equally important.

### How It Works

1. Define the distribution of hyperparameters to sample from
2. Sample random combinations of hyperparameters
3. Train and evaluate a model for each sampled combination
4. Select the combination with the best performance

### Advantages and Disadvantages

#### Advantages
- More efficient than grid search for high-dimensional spaces
- May find better solutions with fewer iterations
- Easily parallelizable

#### Disadvantages
- Not guaranteed to find the optimal solution
- Still requires a well-defined search space
- Like grid search, it's uninformed (doesn't learn from previous iterations)

### Manual Implementation

```python
import numpy as np
from itertools import product
import random

# Create a list of values for hyperparameters
learning_rate_list = list(np.linspace(0.01, 1.5, 200))
min_samples_list = list(range(10, 41))

# Generate all possible combinations
combinations_list = [list(x) for x in product(learning_rate_list, min_samples_list)]

# Sample hyperparameter combinations for a random search
random_combinations_index = np.random.choice(range(0, len(combinations_list)), 250, replace=False)
combinations_random_chosen = [combinations_list[x] for x in random_combinations_index]

print(f"Total possible combinations: {len(combinations_list)}")
print(f"Number of sampled combinations: {len(combinations_random_chosen)}")
print(f"Sample combinations: {combinations_random_chosen[:3]}")

# Alternative approach using random.sample
criterion_list = ['gini', 'entropy']
max_feature_list = ['auto', 'sqrt', 'log2', None]
max_depth_list = list(range(3, 56))

# Generate all possible combinations
combinations_list = [list(x) for x in product(criterion_list, max_feature_list, max_depth_list)]

# Sample hyperparameter combinations using random.sample
combinations_random_chosen = random.sample(combinations_list, 150)

print(f"Total possible combinations: {len(combinations_list)}")
print(f"Number of sampled combinations: {len(combinations_random_chosen)}")
```

### Implementation with Scikit-learn

```python
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from scipy.stats import uniform, randint

# Define the parameter distributions
param_distributions = {
    'learning_rate': uniform(0.01, 0.5),        # Continuous distribution between 0.01 and 0.51
    'n_estimators': randint(50, 500),           # Integer distribution between 50 and 500
    'max_depth': randint(1, 20),                # Integer distribution between 1 and 20
    'min_samples_split': randint(2, 20),        # Integer distribution between 2 and 20
    'min_samples_leaf': randint(1, 20),         # Integer distribution between 1 and 20
    'subsample': uniform(0.5, 0.5),             # Continuous distribution between 0.5 and 1.0
    'max_features': ['auto', 'sqrt', 'log2']    # Categorical distribution
}

# Create a GradientBoostingClassifier
gbm = GradientBoostingClassifier(random_state=42)

# Create RandomizedSearchCV object
random_search = RandomizedSearchCV(
    estimator=gbm,
    param_distributions=param_distributions,
    n_iter=100,                     # Number of parameter settings sampled
    scoring='accuracy',
    cv=5,
    verbose=1,
    random_state=42,
    n_jobs=-1,
    return_train_score=True
)

# Fit to the training data
random_search.fit(X_train, y_train)

# Print the best parameters and score
print("Best parameters:", random_search.best_params_)
print("Best cross-validation score: {:.3f}".format(random_search.best_score_))

# Get the best model
best_gbm = random_search.best_estimator_

# Evaluate on test set
y_pred = best_gbm.predict(X_test)
print("Test set accuracy: {:.3f}".format(accuracy_score(y_test, y_pred)))

# Visualize the hyperparameter distributions sampled
plt.figure(figsize=(12, 8))
for i, param in enumerate(['learning_rate', 'n_estimators', 'max_depth', 'min_samples_leaf']):
    plt.subplot(2, 2, i+1)
    if param in ['learning_rate', 'subsample']:
        plt.hist(random_search.cv_results_[f'param_{param}'])
    else:
        values = random_search.cv_results_[f'param_{param}'].data.astype(float)
        plt.hist(values)
    plt.title(f'Distribution of {param}')
    plt.xlabel(param)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()
```

## Informed Search Methods

Informed search methods use the results of previous evaluations to guide the search for optimal hyperparameters.

### Coarse to Fine Tuning

Coarse to fine tuning is an approach that starts with a broad search and progressively narrows down to promising regions.

#### How It Works

1. **Initial Search**: Start with a random or coarse grid search to identify promising regions
2. **Analysis**: Identify patterns and promising areas in the hyperparameter space
3. **Refinement**: Conduct a more focused search in the promising areas
4. **Iteration**: Repeat the refinement process until optimal performance is achieved

#### Visualization Function for Hyperparameter Analysis

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def visualize_hyperparameter(results_df, param_name, score_col='accuracy'):
    """
    Visualize the relationship between a hyperparameter and model performance.
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        DataFrame containing hyperparameter values and performance metrics
    param_name : str
        Name of the hyperparameter to visualize
    score_col : str, default='accuracy'
        Name of the column containing performance scores
    """
    plt.figure(figsize=(10, 6))
    
    # Create the scatter plot
    plt.scatter(
        results_df[param_name], 
        results_df[score_col], 
        c='blue', 
        alpha=0.6,
        edgecolors='black'
    )
    
    # Add trend line
    z = np.polyfit(results_df[param_name], results_df[score_col], 1)
    p = np.poly1d(z)
    plt.plot(results_df[param_name], p(results_df[param_name]), "r--", alpha=0.8)
    
    # Set plot labels and title
    plt.xlabel(param_name)
    plt.ylabel(score_col)
    plt.title(f'{score_col.capitalize()} for different {param_name} values')
    
    # Set y-axis limits for percentage metrics
    if 'accuracy' in score_col or 'score' in score_col:
        plt.ylim([max(0, results_df[score_col].min() - 0.1), min(1.0, results_df[score_col].max() + 0.1)])
    
    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Show the plot
    plt.tight_layout()
    plt.show()

# Example usage with RandomizedSearchCV results
random_search_results = pd.DataFrame(random_search.cv_results_)

# Visualize the impact of learning_rate
visualize_hyperparameter(
    random_search_results, 
    'param_learning_rate', 
    'mean_test_score'
)

# Visualize the impact of n_estimators
visualize_hyperparameter(
    random_search_results, 
    'param_n_estimators', 
    'mean_test_score'
)
```

![Scatter plot](./assets/coarse-fine-tunning-scatter.png)

#### Implementing Coarse to Fine Tuning

```python
# Step 1: Initial broad random search
param_dist_initial = {
    'max_depth': randint(1, 30),
    'learning_rate': uniform(0.001, 0.999),
    'n_estimators': randint(50, 1000),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 20)
}

initial_search = RandomizedSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_distributions=param_dist_initial,
    n_iter=50,
    cv=3,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)

initial_search.fit(X_train, y_train)

# Step 2: Analyze results to find promising regions
results_df = pd.DataFrame(initial_search.cv_results_)
best_idx = results_df['rank_test_score'] == 1
best_params = results_df.loc[best_idx, 'params'].values[0]

print("Best parameters from initial search:", best_params)

# Step 3: Define refined search space around promising values
refined_max_depth = max(1, best_params['max_depth'] - 5), min(30, best_params['max_depth'] + 5)
refined_learning_rate = max(0.001, best_params['learning_rate'] - 0.1), min(1.0, best_params['learning_rate'] + 0.1)
refined_n_estimators = max(50, best_params['n_estimators'] - 100), min(1000, best_params['n_estimators'] + 100)

param_grid_refined = {
    'max_depth': list(range(refined_max_depth[0], refined_max_depth[1] + 1)),
    'learning_rate': list(np.linspace(refined_learning_rate[0], refined_learning_rate[1], 10)),
    'n_estimators': list(range(refined_n_estimators[0], refined_n_estimators[1] + 1, 25)),
    'min_samples_split': [best_params['min_samples_split']],
    'min_samples_leaf': [best_params['min_samples_leaf']]
}

# Step 4: Perform grid search in the refined space
refined_search = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_grid=param_grid_refined,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

refined_search.fit(X_train, y_train)

# Print results
print("Initial best score: {:.4f}".format(initial_search.best_score_))
print("Refined best score: {:.4f}".format(refined_search.best_score_))
print("Refined best parameters:", refined_search.best_params_)
```

### Bayesian Optimization

Bayesian optimization uses probability models to predict the performance of hyperparameter combinations, focusing evaluations on promising regions.

#### Bayesian Statistics Primer

Bayes' theorem is the foundation of Bayesian optimization:

P(A|B) = (P(B|A) * P(A)) / P(B)

Where:
- P(A|B) is the posterior probability (what we want to find)
- P(B|A) is the likelihood
- P(A) is the prior probability
- P(B) is the evidence

##### Simple Bayesian Calculation Example

```python
# Example: Customer account closure prediction
# Given information:
# - 7% of customers will close their account next month
# - 15% of all customers are unhappy with your product
# - 35% of customers likely to close their account are unhappy

# Assign probabilities to variables 
p_close = 0.07        # Probability of closing account
p_unhappy = 0.15      # Probability of being unhappy
p_unhappy_close = 0.35  # Probability of being unhappy given that they will close

# Calculate: What's the probability that an unhappy customer will close their account?
# Using Bayes' theorem: P(close|unhappy) = P(unhappy|close) * P(close) / P(unhappy)
p_close_unhappy = (p_unhappy_close * p_close) / p_unhappy
print(f"Probability an unhappy customer will close their account: {p_close_unhappy:.4f}")
```

#### Implementation with Hyperopt

[Hyperopt](https://github.com/hyperopt/hyperopt) is a popular library for Bayesian optimization of hyperparameters.

```python
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np

# Define the search space
space = {
    'max_depth': hp.quniform('max_depth', 2, 10, 1),  # Integers between 2 and 10
    'learning_rate': hp.uniform('learning_rate', 0.001, 0.9),  # Continuous between 0.001 and 0.9
    'n_estimators': hp.quniform('n_estimators', 50, 500, 50),  # 50, 100, 150, ..., 500
    'subsample': hp.uniform('subsample', 0.5, 1.0)  # Continuous between 0.5 and 1.0
}

# Define the objective function
def objective(params):
    # Convert some parameters to integers
    params = {
        'max_depth': int(params['max_depth']),
        'n_estimators': int(params['n_estimators']),
        'learning_rate': params['learning_rate'],
        'subsample': params['subsample']
    }
    
    # Create the model with the specified parameters
    gbm = GradientBoostingClassifier(random_state=42, **params)
    
    # Calculate cross-validated accuracy
    score = cross_val_score(gbm, X_train, y_train, scoring='accuracy', cv=5, n_jobs=-1).mean()
    
    # Hyperopt minimizes, so return negative accuracy
    return {
        'loss': -score,  # Negative because hyperopt minimizes
        'status': STATUS_OK,
        'params': params,
        'accuracy': score
    }

# Keep track of the trials
trials = Trials()

# Run the optimization
best = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,  # Tree of Parzen Estimators algorithm
    max_evals=100,  # Number of evaluations
    trials=trials,
    rstate=np.random.RandomState(42)
)

# Convert parameters back to correct types
best_params = {
    'max_depth': int(best['max_depth']),
    'n_estimators': int(best['n_estimators']),
    'learning_rate': best['learning_rate'],
    'subsample': best['subsample']
}

print("Best hyperparameters found:", best_params)

# Get the best score
best_trial_idx = np.argmin([t['result']['loss'] for t in trials.trials])
best_score = -trials.trials[best_trial_idx]['result']['loss']
print(f"Best cross-validation accuracy: {best_score:.4f}")

# Train the final model with the best parameters
final_model = GradientBoostingClassifier(random_state=42, **best_params)
final_model.fit(X_train, y_train)

# Evaluate on the test set
test_accuracy = accuracy_score(y_test, final_model.predict(X_test))
print(f"Test set accuracy: {test_accuracy:.4f}")

# Visualize the optimization process
results = [t['result'] for t in trials.trials]
accuracy_history = [r['accuracy'] for r in results]

plt.figure(figsize=(10, 6))
plt.plot(accuracy_history)
plt.xlabel('Iteration')
plt.ylabel('Cross-validation Accuracy')
plt.title('Hyperopt Optimization Progress')
plt.grid(True)
plt.show()
```

### Genetic Algorithms

Genetic algorithms mimic natural selection to evolve an optimal set of hyperparameters.

#### How It Works

1. **Initial Population**: Generate a random population of hyperparameter sets
2. **Fitness Evaluation**: Evaluate each set's performance
3. **Selection**: Select the best-performing sets
4. **Crossover**: Combine hyperparameters from different sets
5. **Mutation**: Randomly modify some hyperparameters
6. **Iteration**: Repeat the process for multiple generations

## Automated ML Frameworks

Automated Machine Learning (AutoML) frameworks automate the end-to-end process of applying machine learning, including hyperparameter tuning.

### TPOT

[TPOT](http://epistasislab.github.io/tpot/) (Tree-based Pipeline Optimization Tool) is an automated machine learning tool that optimizes machine learning pipelines using genetic programming.

```python
from tpot import TPOTClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Load a sample dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Configure TPOT
tpot = TPOTClassifier(
    generations=5,              # Number of iterations to run the optimization for
    population_size=50,         # Number of individuals in each generation
    offspring_size=25,          # Number of offspring to produce in each generation
    mutation_rate=0.9,          # Mutation rate
    crossover_rate=0.1,         # Crossover rate
    scoring='accuracy',         # Metric to optimize
    cv=5,                       # Cross-validation folds
    verbosity=2,                # Verbosity level
    random_state=42,
    n_jobs=-1,                  # Use all available cores
    max_time_mins=60,           # Maximum time to run in minutes
    config_dict='TPOT sparse'   # Configuration dictionary
)

# Fit TPOT to the data
tpot.fit(X_train, y_train)

# Print the best pipeline
print(tpot.fitted_pipeline_)

# Score on the test set
print(f"Test set accuracy: {tpot.score(X_test, y_test):.4f}")

# Export the optimized pipeline as Python code
tpot.export('tpot_optimized_pipeline.py')
```

## Hyperparameter Tuning Best Practices

1. **Start Simple**: Begin with default hyperparameters and simple models
2. **Use Cross-Validation**: Always use cross-validation to avoid overfitting to the validation set
3. **Prioritize Important Hyperparameters**: Not all hyperparameters have equal impact
4. **Logarithmic Scales**: Use logarithmic scales for hyperparameters with exponential effects (e.g., learning rate)
5. **Monitor Computation Time**: Balance tuning time against expected performance gains
6. **Early Stopping**: Implement early stopping for iterative algorithms to save time
7. **Visualize Results**: Visualize the effect of hyperparameters on performance to gain insights
8. **Ensemble Top Models**: Consider ensembling the top-performing models from your tuning process
9. **Document Your Process**: Keep track of all experiments for reproducibility and knowledge sharing

## Hyperparameters by Algorithm

### Decision Trees

- **max_depth**: Maximum depth of the tree
- **min_samples_split**: Minimum samples required to split a node
- **min_samples_leaf**: Minimum samples required at a leaf node
- **max_features**: Number of features to consider for the best split
- **criterion**: Function to measure split quality ('gini' or 'entropy' for classification)

### Random Forest

- **n_estimators**: Number of trees in the forest
- **max_depth**: Maximum depth of each tree
- **min_samples_split**: Minimum samples required to split a node
- **min_samples_leaf**: Minimum samples required at a leaf node
- **max_features**: Number of features to consider for the best split
- **bootstrap**: Whether to use bootstrap samples

### Gradient Boosting

- **n_estimators**: Number of boosting stages
- **learning_rate**: Shrinks the contribution of each tree
- **max_depth**: Maximum depth of each tree
- **subsample**: Fraction of samples used for fitting each tree
- **min_samples_split**: Minimum samples required to split a node
- **min_samples_leaf**: Minimum samples required at a leaf node
- **max_features**: Number of features to consider for the best split

### Support Vector Machines

- **C**: Regularization parameter
- **kernel**: Kernel type ('linear', 'poly', 'rbf', 'sigmoid')
- **gamma**: Kernel coefficient for 'rbf', 'poly', and 'sigmoid' kernels
- **degree**: Degree of the polynomial kernel
- **class_weight**: Weights associated with classes

### Neural Networks (MLPClassifier/MLPRegressor)

- **hidden_layer_sizes**: Number and size of hidden layers
- **activation**: Activation function ('identity', 'logistic', 'tanh', 'relu')
- **solver**: Weight optimization algorithm ('lbfgs', 'sgd', 'adam')
- **alpha**: L2 regularization term
- **learning_rate**: Learning rate schedule ('constant', 'invscaling', 'adaptive')
- **learning_rate_init**: Initial learning rate
- **max_iter**: Maximum number of iterations
- **batch_size**: Size of minibatches for stochastic optimizers 