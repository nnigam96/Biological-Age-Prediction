
import torch
from constants import *
from tqdm import tqdm
from numpy import vstack
from numpy import sqrt
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch import Tensor
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Module
from torch.optim import SGD
from torch.nn import MSELoss
from torch.nn.init import xavier_uniform_
from constants import *
import numpy as np

class LinearRegression(torch.nn.Module):
    def __init__(self, in_dim=linear_input_dim, out_dim=linear_output_dim):
        super(LinearRegression, self).__init__()

        self.linear = torch.nn.Linear(in_dim, out_dim)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs


def train_model(train_loader, model):
    criterion = MSELoss()
    optimizer = SGD(model.parameters(), lr=linear_lr_rate,
                    momentum=linear_momentum)

    for epoch in range(linear_training_epoch):
        for inputs, targets in tqdm(train_loader):
            optimizer.zero_grad()
            yhat = model(inputs)
            assert not torch.isnan(yhat).any()
            loss = criterion(yhat, targets)
            loss.backward()
            optimizer.step()


def evaluate_model(test_loader, model):
    predictions = []
    actuals = []

    for inputs, targets in tqdm(test_loader):
        yhat = model(inputs)
        yhat = yhat.detach().numpy()
        assert not np.isnan(yhat).any()
        actual = targets.numpy()
        actual = actual.reshape((len(actual), 1))
        predictions.append(yhat)
        actuals.append(actual)

    predictions, actuals = vstack(predictions), vstack(actuals)

    mse = mean_squared_error(actuals, predictions)
    return mse


def linear_regression(train_loader, test_loader, in_dim, out_dim):
    model = LinearRegression(in_dim, out_dim)
    train_model(train_loader, model)
    mse = evaluate_model(test_loader, model)

    print('MSE: %.3f, RMSE: %.3f' % (mse, sqrt(mse)))
    
    
    return model
