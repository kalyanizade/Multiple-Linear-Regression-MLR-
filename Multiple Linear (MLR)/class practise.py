import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r"C:\Users\Kalyani\vscode_projects\spyder\Multiple Linear (MLR)\Investment.csv")

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, 4]

X = pd.get_dummies(X,dtype=int)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)
#y - pred = predict value using x-test data

#== we build mlr model 

m = regressor.coef_
print(m)
# m (slope)(coeffitient)= x train columns = m1+m2+m3+m4+m5++m6

c = regressor.intercept_
print(c)
# c (intercept )

# MLR = [  Y= M1X1 + M2X2 + M3X3 + M4X4 + M5X5 + M6X6 + C ]

X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1) 


import statsmodels.api as sm
X_opt = X[:,[0,1,2,3,4,5]]
#OrdinaryLeastSquares
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()


import statsmodels.api as sm
X_opt = X[:,[0,1,2,3,5]]
#OrdinaryLeastSquares
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()


import statsmodels.api as sm
X_opt = X[:,[0,1,2,3]]
#OrdinaryLeastSquares
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()


import statsmodels.api as sm
X_opt = X[:,[0,1,3]]
#OrdinaryLeastSquares
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()


import statsmodels.api as sm
X_opt = X[:,[0,1]]
#OrdinaryLeastSquares
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()


bias = regressor.score(X_train, y_train)
bias

variance = regressor.score(X_test, y_test)
variance
