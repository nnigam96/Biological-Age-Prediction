
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split

from utils import MSE

from utils import cross_validation


def tree(x, y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)
    export_graphviz(model, out_file='tree.dot')

    MSE(model, X_test, y_test)
    cross_validation(model, x, y)

    return model
