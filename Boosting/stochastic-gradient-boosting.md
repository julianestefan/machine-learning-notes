## Stochastic Gradient Boosting 

* Each tree is trained on a random subset of rows. It also select a random number of features. Then it keeps repeating the process using the residual as target variable in the subsequent models
* add more diversity
* adding further variance to the ensemble of trees

![Gradient Boosting ](./assets/stochastic-gradient-boosting.png)

```python
# Import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error as MSE

# Instantiate sgbr
sgbr = GradientBoostingRegressor(max_depth=4, 
            subsample=0.9,
            max_features=0.75,
            n_estimators=200,
            random_state=2)

# Fit sgbr to the training set
sgbr.fit(X_train, y_train)

# Predict test set labels
y_pred = sgbr.predict(X_test)

# Compute MSE
mse_test = MSE(y_pred, y_test)

# Compute RMSE
rmse_test = mse_test ** (0.5)

# Print RMSE
print('Test set RMSE of gb: {:.3f}'.format(rmse_test))
```