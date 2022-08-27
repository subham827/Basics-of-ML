from this import d
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

diabetes = datasets.load_diabetes()
diabetes_X = diabetes.data[:, np.newaxis, 2]
# print(diabetes_X)
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

model = linear_model.LinearRegression()
model.fit(diabetes_X_train, diabetes_y_train)
diabetes_pred = model.predict(diabetes_X_test)
print("Mean sqared error is: %.2f" % mean_squared_error(diabetes_y_test, diabetes_pred))
print("Variance score is: %.2f" % r2_score(diabetes_y_test, diabetes_pred))
print("Weights" + str(model.coef_))
print("Intercept" + str(model.intercept_))
print("Predicted values: " + str(diabetes_pred))

plt.scatter(diabetes_X_test, diabetes_y_test, color='black')
plt.plot(diabetes_X_test, diabetes_pred, color='blue', linewidth=3)
plt.show()

bcancer = datasets.load_breast_cancer()
bcancer_X = bcancer.data[:, np.newaxis, 2]
bcancer_X_train = bcancer_X[:-20]
bcancer_X_test = bcancer_X[-20:]

bcancer_y_train = bcancer.target[:-20]
bcancer_y_test = bcancer.target[-20:]

model = linear_model.LinearRegression()
model.fit(bcancer_X_train, bcancer_y_train)
bcancer_pred = model.predict(bcancer_X_test)
print("Mean sqared error is: %.2f" % mean_squared_error(bcancer_y_test, bcancer_pred))
print("Variance score is: %.2f" % r2_score(bcancer_y_test, bcancer_pred))
print("Weights" + str(model.coef_))
print("Intercept" + str(model.intercept_))
print("Predicted values: " + str(bcancer_pred))

plt.scatter(bcancer_X_test, bcancer_y_test, color='black')
plt.plot(bcancer_X_test, bcancer_pred, color='red', linewidth=3)
plt.show()

