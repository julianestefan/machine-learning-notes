Unsupervised no labeled data/ It finds the patterns 

Clustering group records with similar characteristics

Preliminary analysis suing scatter plot

### Hierarchical


```python
# Import linkage and fcluster functions
from scipy.cluster.hierarchy import linkage, fcluster

# Use the linkage() function to compute distance
Z = linkage(df, 'ward')

# Generate cluster labels
df['cluster_labels'] = fcluster(Z, 2, criterion='maxclust')

# Plot the points with seaborn
sns.scatterplot(x="x", y="y", hue="cluster_labels", data=df)
plt.show()
```

### K means

Calculate random points. After that calculate the distance to points and recompute to be the mean of the group. This process is repeated an arbitrary numbers of times.


```python
# Import kmeans and vq functions
from scipy.cluster.vq import kmeans, vq

# Compute cluster centers
centroids,_ = kmeans(df, 2)

# Assign cluster labels
df['cluster_labels'], _ = vq(df, centroids)

# Plot the points with seaborn
sns.scatterplot(x="x", y="y", hue="cluster_labels", data=df)
plt.show()
```

## Normalization 

is needed to keep variables in a similar scale.

``` python
# Import the whiten function
from scipy.cluster.vq import whiten

goals_for = [4,3,2,3,1,1,2,0,1,4]

# Use the whiten() function to standardize the data
scaled_data = whiten(goals_for)
print(scaled_data)

# Plot original data
plt.plot(goals_for, label='original')

# Plot scaled data
plt.plot(scaled_data, label='scaled')

# Show the legend in the plot
plt.legend()

# Display the plot
plt.show()
```