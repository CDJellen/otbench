from typing import Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from otbench.benchmark.models.forecasting.pytorch.base_pytorch_model import BasePyTorchForecastingModel


class CNN(nn.Module):

    def __init__(self,
                 input_size: int,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 kernel_size: Union[Tuple[int, int], int] = 1,
                 stride: Union[Tuple[int, int], int] = 1,
                 padding: Union[Tuple[int, int], int, str] = "valid",
                 bias: bool = True):
        super(CNN, self).__init__()
        self.input_size = input_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        if input_size <= 0:
            raise AssertionError("input size must be greater than 0")
        if kernel_size > input_size:
            raise AssertionError("kernel size must be less than the number of features")

        self.cnn = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear((input_size - kernel_size + 1), 1)

    def forward(self, x):
        out = self.cnn(x)
        out = self.relu(out)
        out = self.fc(out)
        return out


class CNNModel(BasePyTorchForecastingModel):
    """A basic PyTorch CNN model."""

    def __init__(self,
                 name: str,
                 window_size: int,
                 forecast_horizon: int,
                 target_name: str,
                 input_size: int,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 kernel_size: Union[Tuple[int, int], int] = 1,
                 stride: Union[Tuple[int, int], int] = 1,
                 padding: Union[Tuple[int, int], int, str] = "valid",
                 bias: bool = True,
                 batch_size: int = 32,
                 n_epochs: int = 500,
                 learning_rate: float = 0.025,
                 criterion: 'torch.nn.modules.loss' = nn.MSELoss(),
                 optimizer: 'torch.optim' = optim.SGD,
                 normalize_data: bool = True,
                 random_state: int = 2020,
                 verbose: bool = False,
                 **kwargs):
        super().__init__(name=name,
                         window_size=window_size,
                         forecast_horizon=forecast_horizon,
                         target_name=target_name,
                         batch_size=batch_size,
                         n_epochs=n_epochs,
                         learning_rate=learning_rate,
                         criterion=criterion,
                         optimizer=optimizer,
                         random_state=random_state,
                         verbose=verbose)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

        # create and set the model
        model = CNN(input_size, in_channels, out_channels, kernel_size, stride, padding, bias)
        self.set_model(model=model, normalize_data=normalize_data,
                       set_optimizer_callable_params=True)  # apply model params to SGD

    def train(self, X: 'pd.DataFrame', y: 'pd.DataFrame'):
        # maintain the same interface as the other models
        n_features = len(X.columns) // self.window_size
        if self.verbose:
            print(f"training data contains {n_features} features.")
        # set train dataloader
        self.set_training_data(X=X, y=y)
        # train the model
        torch.manual_seed(self.random_state)
        for i in range(self.n_epochs):
            for _, (X, y) in enumerate(self.train_dataloader):
                self.optimizer.zero_grad()
                outputs = self.model(X.float())
                loss = self.criterion(outputs, y.float())
                loss.backward()
                self.optimizer.step()
            if self.verbose and self.n_epochs >= 10 and (i % (self.n_epochs // 10) == 0):
                print(f"at epoch {i}. loss: {loss}")

    def predict(self, X: 'pd.DataFrame'):
        """Generate predictions from the RNNModel."""
        n_features = len(X.columns) // self.window_size
        if self.verbose:
            print(f"validation data contains {n_features} features.")
        y = X.iloc[:, [0]]
        self.set_validation_data(X=X, y=y)

        pred = []
        with torch.no_grad():
            for _, (X, _) in enumerate(self.val_dataloader):
                y_pred = self.model(X.float())
                if self.normalize_data:
                    y_pred = y_pred * self.y_std + self.y_mean
                y_pred = y_pred.numpy()

                # add the prediction value to the list
                pred.append(y_pred[0][0])

        return np.array(pred)
