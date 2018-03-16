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
#Import the optimal model using Backward elimination
import statsmodels.api as sm



# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding dummy variable trap
X=X[:,1:]

#creating the splitting 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Training your machine learning
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#predicting  the Test set results
y_pred = regressor.predict(X_test)

#Creating the optimal model using backward elimination
#We need to add the b0 of our poliminial to use the backward elimination and calculate the p-value. To 
#avoid problems with math, we will add b0 as value 1 in all rows 
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)

#using a new variable only to see help us to calculate de p-value 
X_opt = X[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
print (regressor_OLS.summary())

X_opt = X[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
print (regressor_OLS.summary())

X_opt = X[:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
print (regressor_OLS.summary())

X_opt = X[:,[0,3,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
print (regressor_OLS.summary())

