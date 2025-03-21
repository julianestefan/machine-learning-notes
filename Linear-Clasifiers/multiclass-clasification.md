
Onve-vs-rest

* Creating a model to each class. 
* Predict with all take the largest output

``` python
lr_ovr = LogisticRegression(multi_class='ovr')
```

Multinomial/softmax
* Fit a single classifier for all classes
* prediction directly outputs best class


``` python
lr_mn = LogisticRegression(multi_class='multinomial')
```

