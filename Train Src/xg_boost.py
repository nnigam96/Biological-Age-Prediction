import xgboost as xgb
from utils import cross_validation
from sklearn.model_selection import train_test_split

from utils import MSE


def xgb_regression(x, y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    model = xgb.XGBRegressor(
        n_estimators=100, max_depth=2, eta=0.1, subsample=0.7, colsample_bytree=0.8)
    model.fit(X_train, y_train)

    MSE(model, X_test, y_test)
    cross_validation(model, x, y)

    return model
