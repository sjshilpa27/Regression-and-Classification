import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

data=pd.DataFrame(np.loadtxt("linear-regression.txt",delimiter=","))

feature_names = [0,1]
X = data[feature_names]
y = data[2]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#calling Regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train) #training the algorithm
print("Retrieve the intercept")
print(regressor.intercept_)
print("Retrieving the slope")
print(regressor.coef_)
y_pred = regressor.predict(X_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df1 = df.head(25)
df1.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

ones = np.ones((X.shape[0],1))
plot_x = np.hstack((ones,X))



