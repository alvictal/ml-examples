# Importing the libraries
# Math libraries
import numpy as np
# Import and manage datasets
import pandas as pd
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#import Multiple Linear REgression 
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# Importing the dataset
dataset = pd.read_csv('estats2.csv')
x = dataset.iloc[:-1, 1:-1].values
y = dataset.iloc[:-1, 4].values


print (x)
print (y)

# Encoding categorical data
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Avoiding dummy variable trap
#X=X[:,1:]

#Training your machine learning
x = np.append(arr = np.ones((1636,1)).astype(int), values = x, axis = 1)

#creating the splitting 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.01, random_state = 0)

#regressor_OLS = sm.OLS(endog = y, exog = x).fit()
#print (regressor_OLS.summary())

regressor = LinearRegression()
regressor.fit(x_train,y_train)

#predicting  the Test set results
#y_pred = regressor.predict([[1,1000,1,192]])
y_pred = regressor.predict(x_test)
print (x_test)
print (y_test)
print (y_pred)

