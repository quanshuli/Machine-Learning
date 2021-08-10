# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 11:02:56 2021

@author: peace
"""

import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


#%% 1
# linear regression
rng = np.random.RandomState(42)
x = 10 * rng.rand(50)
y = 2 * x - 1 + rng.randn(50)

model = LinearRegression(fit_intercept=True)
X = x[:, np.newaxis]
model.fit(X, y)

xfit = np.linspace(-1, 11)
Xfit = xfit[:, np.newaxis]
yfit = model.predict(Xfit)

plt.scatter(x, y)
plt.plot(xfit, yfit)

#%% 2
# Supervised learning example: Iris classification
# arrange data
from sklearn.model_selection import train_test_split

iris = sns.load_dataset('iris')
X_iris = iris.drop('species',axis=1)
y_iris = iris['species']
Xtrain, Xtest, ytrain, ytest = train_test_split(X_iris, y_iris,
                                                random_state=1)

#sns_plot = sns.pairplot(iris, hue='species', height=1.5)
#sns_plot.savefig('iris.png')
#%% 3
# use gaussian naive bayes model to predict
from sklearn.naive_bayes import GaussianNB # 1. choose model class
model = GaussianNB()                       # 2. instantiate model
model.fit(Xtrain, ytrain)                  # 3. fit model to data
y_model = model.predict(Xtest)             # 4. predict on new data

#%% 4
# Finally, we can use the accuracy_score utility to see the fraction of predicted labels that match their true value

from sklearn.metrics import accuracy_score
accuracy_score(ytest, y_model)

#%% 5
# unsupervised example
from sklearn.decomposition import PCA  # 1. Choose the model class
model = PCA(n_components=2)            # 2. Instantiate the model with hyperparameters
model.fit(X_iris)                      # 3. Fit to data. Notice y is not specified!
X_2D = model.transform(X_iris)         # 4. Transform the data to two dimensions

iris['PCA1'] = X_2D[:, 0]
iris['PCA2'] = X_2D[:, 1]
sns.lmplot("PCA1", "PCA2", hue='species', data=iris, fit_reg=False)
#%% 6
# gaussian mixture model
from sklearn import mixture      # 1. Choose the model class
model = mixture.GaussianMixture(n_components=3,
            covariance_type='full')  # 2. Instantiate the model with hyperparameters
model.fit(X_iris)                    # 3. Fit to data. Notice y is not specified!
y_gmm = model.predict(X_iris)        # 4. Determine cluster labels

iris['cluster'] = y_gmm
sns.lmplot("PCA1", "PCA2", data=iris, hue='species',
           col='cluster', fit_reg=False)


#%% 7
# explore hand-written digits
from sklearn.datasets import load_digits
digits = load_digits()
digits.images.shape

X = digits.data
y = digits.target

# Unsupervised learning: Dimensionality reduction
from sklearn.manifold import Isomap
iso = Isomap(n_components=2)
iso.fit(digits.data)
data_projected = iso.transform(digits.data)

# Classification on digits
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=0)

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(Xtrain, ytrain)
y_model = model.predict(Xtest)

from sklearn.metrics import accuracy_score
accuracy_score(ytest, y_model)

from sklearn.metrics import confusion_matrix
mat = confusion_matrix(ytest, y_model)

sns.heatmap(mat, square=True, annot=True, cbar=False)
plt.xlabel('predicted value')
plt.ylabel('true value')




























