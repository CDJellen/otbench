import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from otbench.benchmark.models.forecasting.pytorch.base_pytorch_model import BasePyTorchForecastingModel


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
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device).requires_grad_()
        out, _ = self.rnn(x, h0.detach())
        out = self.fc(out[:, -1, :])
        return out


class RNNModel(BasePyTorchForecastingModel):
    """A basic PyTorch RNN model."""

    def __init__(self,
                 name: str,
                 window_size: int,
                 forecast_horizon: int,
                 target_name: str,
                 input_size: int,
                 hidden_size: int = 512,
                 num_layers: int = 2,
                 num_classes: int = 1,
                 output_size: int = None,
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
                         verbose=verbose,
                         **kwargs)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = output_size if output_size is not None else num_classes

        # create and set the model
        model = RNN(input_size, hidden_size, num_layers, self.num_classes)
        self.set_model(model=model, normalize_data=normalize_data,
                       set_optimizer_callable_params=True)  # apply model params to SGD

    def _train(self, X: 'pd.DataFrame', y: 'pd.DataFrame'):
        # Guard against empty data
        if len(X) == 0:
            return

        # 1. Calculate actual features per timestep in the data.
        # Matches reshape logic in _set_dataloader_from_data:
        # temporal mode when columns divide evenly, flat mode otherwise.
        n_cols = len(X.columns)
        if self.window_size > 1 and n_cols % self.window_size == 0:
            n_features_in_data = n_cols // self.window_size
        else:
            n_features_in_data = n_cols  # flat/single-step mode

        # 2. Validate against initialized architecture
        if n_features_in_data != self.input_size:
            raise ValueError(f"Dimension Mismatch: Model initialized with input_size={self.input_size}, "
                             f"but training data has {n_features_in_data} features per timestep "
                             f"(Total columns: {len(X.columns)}, Window: {self.window_size}).")

        if self.verbose:
            mode = "temporal" if (self.window_size > 1 and n_cols % self.window_size == 0) else "flat"
            print(f"training data: {mode} mode, {n_features_in_data} features per timestep, "
                  f"{self.window_size} timestep(s) per sample.")

        # 3. Proceed with standard training
        self.set_training_data(X=X, y=y)

        torch.manual_seed(self.random_state)
        for i in range(self.n_epochs):
            for _, (X_batch, y_batch) in enumerate(self.train_dataloader):
                self.optimizer.zero_grad()
                X_batch, y_batch = X_batch.to(self.device).float(), y_batch.to(self.device).float()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()

            if self.verbose and self.n_epochs >= 10 and (i % (self.n_epochs // 10) == 0):
                print(f"at epoch {i}. loss: {loss.item():.6f}")

    def _predict(self, X: 'pd.DataFrame'):
        """Generate predictions from the RNNModel."""
        if len(X) == 0:
            return np.empty((0, self.num_classes))

        n_cols = len(X.columns)
        n_features = n_cols // self.window_size if (self.window_size > 1 and n_cols % self.window_size == 0) else n_cols
        if self.verbose:
            mode = "temporal" if (self.window_size > 1 and n_cols % self.window_size == 0) else "flat"
            print(f"validation data: {mode} mode, {n_features} features per timestep.")
        self.set_validation_data(X=X, y=None)

        pred = []
        with torch.no_grad():
            for _, (X, _) in enumerate(self.val_dataloader):
                X = X.to(self.device).float()
                y_pred = self.model(X)
                if self.normalize_data:
                    # y_std and y_mean are numpy arrays, need to move y_pred to cpu
                    y_pred = y_pred.cpu()
                    y_pred = y_pred * self.y_std + self.y_mean

                y_pred = y_pred.cpu().numpy()

                # add the prediction value to the list
                if self.num_classes == 1:
                    pred.append(y_pred[0][0])
                else:
                    pred.append(y_pred[0])

        return np.array(pred)