# K-Nearest Neighbors (K-NN)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
def calcProbability(file_name, acc, vsvezes): 
	dataset = pd.read_csv(file_name)
	X = dataset.iloc[1:, 1:3].values
	y = dataset.iloc[1:, 3].values

	# Feature Scaling
	from sklearn.preprocessing import StandardScaler
	sc = StandardScaler()
	X = sc.fit_transform(X)
	
	
	# Fitting K-NN to the Training set
	from sklearn.neighbors import KNeighborsClassifier
	classifier = KNeighborsClassifier(n_neighbors = 5, weights = 'distance', metric = 'minkowski', p = 2)
	classifier.fit(X, y)

	# Predicting the Test set results
	x_test = [[vsvezes,180]]
	for i in range(200-181):
		x_test = x_test + [[vsvezes,181+i]]

	x_test = sc.transform(x_test)
	y_pred = classifier.predict(x_test)
	
	# Making the Confusion Matrix
	#from sklearn.metrics import confusion_matrix
	#cm = confusion_matrix(y, y_pred)

	yes = 0
	for i in y_pred:
		if i == 1:
			yes = yes + 1

	return ((yes/len(y_pred))*100)




pred = [[1],
		[1],
		[0],
		[1],
		[0],
		[0],
		[0],
		[1],
		[3],
		[0],
		[1],
		[0],
		[1],
		[0],
		[0],
		[1],
		[0],
		[0],
		[1],
		[0],
		[4],
		[0],
		[0],
		[0],
		[0]]


d = dict()
for i in range(1,26):
	file_name = str(i) + ".csv"
	
	result = calcProbability(file_name, pred[i-1][0],pred[i-1][0])
	d[i] = float("%.2f" % result)
	
d_order = sorted(d.items(), key=lambda kv: kv[1])

valuesum = 0
for i in range(15):
	print(d_order[24-i])
	valuesum = valuesum + float(d_order[24-i][0])

print ("Probability sum: " + str(valuesum))
print (d_order)
	
