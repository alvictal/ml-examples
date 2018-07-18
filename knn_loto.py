# K-Nearest Neighbors (K-NN)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('1.csv')
X = dataset.iloc[1:, 0:3].values
y = dataset.iloc[1:, 3].values

print(X)
print(y)
# Splitting the dataset into the Training set and Test set

# Feature Scaling
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X_train = sc.fit_transform(X)
#print(X_train)

# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 1)
classifier.fit(X, y)

# Predicting the Test set results

x_test = [[1023,0,166]]
for i in range(226-167):
	x_test = x_test + [[1023,0,167+i]]
print(x_test)
#x_test = sc.fit_transform(x_test)
#print(x_test)
y_pred = classifier.predict(x_test)

print(y_pred)
