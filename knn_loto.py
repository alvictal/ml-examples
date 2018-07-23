# K-Nearest Neighbors (K-NN)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
def calcProbability(file_name, acc, vsvezes): 
	dataset = pd.read_csv(file_name)
	X = dataset.iloc[1:, 0:3].values
	y = dataset.iloc[1:, 3].values

	# Fitting K-NN to the Training set
	from sklearn.neighbors import KNeighborsClassifier
	classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 1)
	classifier.fit(X, y)

	# Predicting the Test set results
	x_test = [[acc,vsvezes,166]]
	for i in range(226-167):
		x_test = x_test + [[acc,vsvezes,167+i]]

	y_pred = classifier.predict(x_test)

	yes = 0
	for i in y_pred:
		if i == 1:
			yes = yes + 1

	return ((yes/len(y_pred))*100)




pred = [[1025,0],
		[1038,0],
		[1027,0],
		[1025,0],
		[1015,0],
		[986,2],
		[994,0],
		[961,0],
		[1005,1],
		[1034,0],
		[1047,0],
		[1011,0],
		[1048,0],
		[1025,2],
		[1010,0],
		[963,0],
		[1013,2],
		[1009,0],
		[1007,0],
		[1029,1],
		[1000,3],
		[1014,1],
		[1014,1],
		[1041,3],
		[1024,1]]


d = dict()
for i in range(1,26):
	file_name = str(i) + ".csv"
	
	result = calcProbability(file_name, pred[i-1][0],pred[i-1][1])
	d[i] = "%.2f" % result
	
d_order = sorted(d.items(), key=lambda kv: kv[1])

valuesum = 0
for i in range(15):
	print(d_order[24-i])
	valuesum = valuesum + float(d_order[24-i][0])

print ("Probability sum: " + str(valuesum))
print (d_order)
	
