
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from utils import MSE

import matplotlib.pyplot as plt
from utils import cross_validation

def linear(x, y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    model = LinearRegression()
    model.fit(X_train, y_train)

    MSE(model, X_test, y_test)
    cross_validation(model, x, y)

    return model
