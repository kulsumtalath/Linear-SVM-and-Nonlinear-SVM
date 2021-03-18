# LINEAR SVM


```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
```


```python
X = np.array([1, 5, 1.5, 8, 1, 9, 7, 8.7, 2.3, 5.5, 7.7, 6.1])
y = np.array([2, 8, 1.8, 8, 0.6, 11, 10, 9.4, 4, 3, 8.8, 7.5])
```


```python
plt.scatter(X, y)  #show unclassified data
plt.show()
```


![png](output_3_0.png)



```python
# shaping data for training the model, pre-processing on the already structured code. 
#This will put the raw data into a format that we can use to train the SVM model.
training_X = np.vstack((X, y)).T
training_y = [0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1]
```


```python
#Now we can create the SVM model using a linear kernel.
# define the model
clf = svm.SVC(kernel='linear', C=1.0)
```


```python
# train the model
clf.fit(training_X, training_y)
```




    SVC(kernel='linear')




```python
# get the weight values for the linear equation from the trained SVM model
w = clf.coef_[0]

# get the y-offset for the linear equation
a = -w[0] / w[1]

# make the x-axis space for the data points
XX = np.linspace(0, 13)

# get the y-values to plot the decision boundary
yy = a * XX - clf.intercept_[0] / w[1]

# plot the decision boundary
plt.plot(XX, yy, 'k-')

# show the plot visually
plt.scatter(training_X[:, 0], training_X[:, 1], c=training_y)
plt.legend()
plt.show()
```

    No handles with labels found to put in legend.
    


![png](output_7_1.png)


# NON-LINEAR SVM


```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn import svm
```


```python
# non-linear data
circle_X, circle_y = datasets.make_circles(n_samples=300, noise=0.05)
```


```python
# show raw non-linear data
plt.scatter(circle_X[:, 0], circle_X[:, 1], c=circle_y, marker='.')
plt.show()
```


![png](output_11_0.png)


Now that you can see how the data are separated, we can choose a non-linear SVM to start with. 
This dataset doesn't need any pre-processing before we use it to train the model,
so we can skip that step. 
Here's how the SVM model will look for this:


```python
# make non-linear algorithm for model
nonlinear_clf = svm.SVC(kernel='rbf', C=1.0)
```


```python
# training non-linear model
nonlinear_clf.fit(circle_X, circle_y)
```




    SVC()




```python
# Plot the decision boundary for a non-linear SVM problem
def plot_decision_boundary(model, ax=None):
    if ax is None:
        ax = plt.gca()
        
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)

	# shape data
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    
	# get the decision boundary based on the model
    P = model.decision_function(xy).reshape(X.shape)
    
    # plot decision boundary
    ax.contour(X, Y, P,
               levels=[0], alpha=0.5,
               linestyles=['-'])
```


```python
plt.scatter(circle_X[:, 0], circle_X[:, 1], c=circle_y, s=50)
plot_decision_boundary(nonlinear_clf)
plt.scatter(nonlinear_clf.support_vectors_[:, 0], nonlinear_clf.support_vectors_[:, 1], s=50, lw=1, facecolors='none')
plt.show()
```


![png](output_16_0.png)



```python

```
