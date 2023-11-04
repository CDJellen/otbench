from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from otb.benchmark.models.regression.base_model import BaseRegressionModel


class RNNModel(BaseRegressionModel):
    """A basic PyTorch RNN model."""

    def __init__(
        self,
        name: str,
        input_size: int,
        obs_window_size: int = 1,
        hidden_size: int = 512,
        num_layers: int = 2,
        num_classes: int = 1,
        batch_size: int = 32,
        n_epochs: int = 500,
        learning_rate: float = 0.025,
        criterion: 'torch.nn.modules.loss' = nn.MSELoss(),
        optimizer: 'torch.optim' = optim.Adam,
        random_state: int = 2020,
        **kwargs
    ):
        self.name = name

        # persist the mean and std of the training data
        self.X_mean = None
        self.X_std = None
        self.y_mean = None
        self.y_std = None
        self.obs_window_size = obs_window_size

        # create the model
        self.batch_size = batch_size
        self.random_state = random_state
        self.model = RNN(input_size, hidden_size, num_layers, num_classes)
        self.n_epochs = n_epochs
        self.criterion = criterion
        self.optimizer = optimizer(self.model.parameters(), lr=learning_rate)
    
    def train(self, X: 'pd.DataFrame', y: 'pd.DataFrame'):
        # maintain the same interface as the other models
        n_features = len(X.columns)

        # map X and y to numpy arrays
        X = X.to_numpy()
        y = y.to_numpy()

        # normalize the training data
        X_mean = np.nanmean(X, axis=0)
        X_std = np.nanstd(X, axis=0)
        y_mean = np.nanmean(y, axis=0)
        y_std = np.nanstd(y, axis=0)

        # save the mean and std
        self.X_mean = X_mean
        self.X_std = X_std
        self.y_mean = y_mean
        self.y_std = y_std

        # replace missing values with the mean of that column
        X[np.isnan(X)] = np.take(X_mean, np.where(np.isnan(X))[1])
        y[np.isnan(y)] = np.take(y_mean, np.where(np.isnan(y))[1])

        # normalize the data before training
        X = (X - X_mean) / X_std
        y = (y - y_mean) / y_std

        # convert to torch tensors
        X = torch.from_numpy(X)
        y = torch.from_numpy(y)

        # create the dataset and dataloader
        train_dataset = torch.utils.data.TensorDataset(X, y)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # train the model
        torch.manual_seed(self.random_state)
        for _ in range(self.n_epochs):
            for _, (X, y) in enumerate(train_loader):
                X = X.reshape(-1, self.obs_window_size, n_features)  # correct for shape
                self.optimizer.zero_grad()
                outputs = self.model(X.float())
                loss = self.criterion(outputs, y.float())
                loss.backward()
                self.optimizer.step()

    def predict(self, X: 'pd.DataFrame'):
        # predict the mean for each entry in X
        n_features = len(X.columns)
        
        # map X to numpy array
        X = X.to_numpy()

        # replace missing values with the mean of that column
        X[np.isnan(X)] = np.take(self.X_mean, np.where(np.isnan(X))[1])
        
        X = (X - self.X_mean) / self.X_std
        y = np.ones((X.shape[0], 1)) * self.y_mean
        X = torch.from_numpy(X)
        y = torch.from_numpy(y)

        test_dataset = torch.utils.data.TensorDataset(X, y)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
        
        pred = []

        with torch.no_grad():
            for _, (X, _) in enumerate(test_loader):
                X = X.reshape(-1, self.obs_window_size, n_features)
                y_pred = self.model(X.float())
                y_pred = y_pred * self.y_std + self.y_mean
                y_pred = y_pred.numpy()

                # add the prediction value to the list
                pred.append(y_pred[0][0])
        
        return np.array(pred)


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        out, _ = self.rnn(x, h0.detach())
        out = self.fc(out[:, -1, :])
        return out
