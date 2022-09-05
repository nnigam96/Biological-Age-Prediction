from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import train_test_split

from utils import MSE

from utils import cross_validation


def knn(x, y, n=4):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    model = KNeighborsRegressor(n_neighbors=n)
    model.fit(X_train, y_train)

    MSE(model, X_test, y_test)
    cross_validation(model, x, y)

    return model
