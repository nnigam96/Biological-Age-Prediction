import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

from numpy import vstack
from numpy import sqrt
import pandas as pd

from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch
from torch import Tensor
from torch.nn import Linear
from torch.nn import Sigmoid
from torch.nn import Module
from torch.optim import SGD
from torch.nn import MSELoss
from torch.nn.init import xavier_uniform_

from constants import *
from skimpy import skim, clean_columns
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold


def clean_data(df):
    df = df.drop(df.filter(regex='Unnamed').columns, axis=1)
    df = clean_columns(df, case="camel")

    # TODO: Fix NaN based on the column
    # remove rows that don't have entries
    df = df.dropna(subset=NAN_CT_COLUMNS + NAN_CLINCAL_COLUMNS)

    # df=df.dropna(subset=['deathDFromCt'])

    return df


def convert(df, normalize=False, augmentDeath=True):
    labelencoder = LabelEncoder()
    min_max_scaler = preprocessing.MinMaxScaler()
    labelbinarizer = LabelBinarizer()

    for col in df.columns:

        if col in CATEGORICAL:
            # df[col].fillna(df[col].mode(), inplace=True)
            df[col] = labelencoder.fit_transform(df[col])
        elif col in CATEGORICAL_OUTCOMES:
            idx = ~df[col].isna()
            df.loc[idx, col] = np.sum(labelbinarizer.fit(
                df.loc[idx, col]).transform(df.loc[idx, col]), axis=1)
            df[col].fillna(0, inplace=True)
        else:
            if col == 'deathDFromCt':
                 df[col].fillna(0, inplace=True)
            
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # df[col].fillna(df[col].mean(), inplace=True)
    if normalize:
        df = min_max_scaler.fit_transform(df)

    df = np.nan_to_num(df)

    # df[np.isnan(df)] = 0
    assert not np.isnan(df).any()
    return df


class ExcelDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        inputs = self.X[idx]
        targets = self.y[idx]
        return inputs, targets

    def get_splits(self, n_test=TEST_SPLIT, n_validation=VALIDATION_SPLIT):
        test_size = round(n_test * len(self.X))
        validation_size = round(n_validation * len(self.X))
        train_size = len(self.X) - test_size - validation_size
        r = random_split(self, [train_size, test_size, validation_size])
        return r


def get_loaders(data):
    train, test, validation = data.get_splits()

    train_loader = DataLoader(
        train, batch_size=PERCEPTRON_TRAIN_BATCH_SIZE, shuffle=True)

    test_loader = DataLoader(
        test, batch_size=PERCEPTRON_TEST_BATCH_SIZE, shuffle=False)

    validation_loader = DataLoader(
        validation, batch_size=PERCEPTRON_TEST_BATCH_SIZE, shuffle=False)

    return train_loader, test_loader, validation_loader


def load_data(data_path, normalizeCT=True, normalizeClinical=False, augmentDeath=True):
    df = pd.read_excel(data_path)
    df = clean_data(df)

    # ct data
    ct_data = df.loc[:, 'l1HuBmd':'liverHuMedian']
    ct_data = convert(ct_data, normalize=normalizeCT)
    ct_data = Tensor(ct_data)
    assert not torch.isnan(ct_data).any()

    # clinical data
    clinical_data = df.loc[:, 'clinicalFUIntervalDFromCt':'metSx']
    clinical_data = convert(clinical_data, normalize=normalizeClinical)
    clinical_data = Tensor(clinical_data)
    assert not torch.isnan(clinical_data).any()

    # outcomes data
    outcomes_data = df.loc[:, 'deathDFromCt':'primaryCancerSite2DxDFromCt']
    outcomes_data = convert(outcomes_data, augmentDeath=augmentDeath)
    outcomes_data = Tensor(outcomes_data)
    assert not torch.isnan(outcomes_data).any()

    data = df
    # assert not torch.isnan(data).any()

    return ct_data, clinical_data, outcomes_data


def cross_validation(model, x, y):
    cv = RepeatedKFold(n_splits=10, n_repeats=3)

    scores = cross_val_score(
        model, x, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)

    print('Mean -ve Absolute Error: %.3f (%.3f)' %
          (scores.mean(), scores.std()))


# def confusion_matrix():
#     cnf_matrix = metrics.confusion_matrix(y_test, y_pred)


# def roc():
#     y_pred_proba = logreg.predict_proba(X_test)[::, 1]
#     fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
#     auc = metrics.roc_auc_score(y_test, y_pred_proba)
#     plt.plot(fpr, tpr, label="data 1, auc="+str(auc))
#     plt.legend(loc=4)
#     plt.show()

def MSE(model, X_test, y_test):

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    print("MSE: ", mse)
    print("RMSE: ", mse**(1/2.0))


def FetchAugmentedDeath(cfu, alpha, death):
     # TODO: Add code here to fill Death values
    aug_death=death
    max_death=max(death)
    for i,d in enumerate(death):
        if d==0:
            aug_death[i]=cfu[i]+((cfu[i]+max_death)*0.5*(1-alpha[i])) 
        else:
            continue    
    return aug_death

