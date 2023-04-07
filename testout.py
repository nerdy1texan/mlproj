from sklearn.impute import SimpleImputer
import numpy as np

#creating a 2Dnumpy array with some missing valuyes
X = np.array([[1, 2, np.nan], [3, np.nan, 5],[np.nan, 7, 8]])
print(X)

#createa simple inputer using mean strategy
imputer = SimpleImputer(strategy='mean')
print(imputer)

#fit the imputer to thedata and transform the data
X_imputed = imputer.fit_transform(X)
print(X_imputed)
Y = (5+8)/2
print(Y)