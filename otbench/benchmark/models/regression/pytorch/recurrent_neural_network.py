import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from otbench.benchmark.models.regression.pytorch.base_pytorch_model import BasePyTorchRegressionModel


class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        if input_size <= 0:
            raise AssertionError("input size must be greater than 0")

        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        out, _ = self.rnn(x, h0.detach())
        out = self.fc(out[:, -1, :])
        return out


class RNNModel(BasePyTorchRegressionModel):
    """A basic PyTorch RNN model."""

    def __init__(
            self,
            name: str,
            target_name: str,
            input_size: int,
            window_size: int = 1,  # default to single row
            hidden_size: int = 512,
            num_layers: int = 2,
            num_classes: int = 1,
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
                         target_name=target_name,
                         window_size=window_size,
                         batch_size=batch_size,
                         n_epochs=n_epochs,
                         learning_rate=learning_rate,
                         criterion=criterion,
                         optimizer=optimizer,
                         random_state=random_state,
                         verbose=verbose)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes

        # create and set the model
        model = RNN(input_size, hidden_size, num_layers, num_classes)
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
        self.set_validation_data(X=X, y=None)

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
