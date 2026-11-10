import torch.nn as nn
import torch.optim as optim

from otbench.benchmark.models._pytorch_base import RNN
from otbench.benchmark.models.forecasting.pytorch.base_pytorch_model import BasePyTorchForecastingModel


class RNNModel(BasePyTorchForecastingModel):
    """A configurable PyTorch recurrent model supporting RNN, LSTM, and GRU cells."""

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
                 cell_type: str = 'rnn',
                 batch_size: int = 32,
                 n_epochs: int = 500,
                 learning_rate: float = 0.025,
                 criterion: 'torch.nn.modules.loss' = nn.MSELoss(),
                 optimizer: 'torch.optim' = optim.SGD,
                 normalize_data: bool = True,
                 random_state: int = 2020,
                 verbose: bool = False,
                 weight_decay: float = 0.0,
                 patience: int = None,
                 val_fraction: float = 0.0,
                 use_scheduler: bool = False,
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
                         weight_decay=weight_decay,
                         patience=patience,
                         val_fraction=val_fraction,
                         use_scheduler=use_scheduler,
                         **kwargs)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = output_size if output_size is not None else num_classes
        self.cell_type = cell_type

        model = RNN(input_size, hidden_size, num_layers, self.num_classes, cell_type=cell_type)
        self.set_model(model=model, normalize_data=normalize_data, set_optimizer_callable_params=True)

    def _train(self, X: 'pd.DataFrame', y: 'pd.DataFrame'):
        self._run_training_loop(X=X, y=y, input_size=self.input_size)

    def _predict(self, X: 'pd.DataFrame'):
        """Generate predictions from the RNNModel."""
        return self._run_prediction(X=X, output_dim=self.num_classes)


class GRUModel(RNNModel):
    """A Gated Recurrent Unit (GRU) forecasting model compatible with otbench."""

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
                 cell_type: str = 'gru',
                 batch_size: int = 32,
                 n_epochs: int = 100,
                 learning_rate: float = 0.001,
                 criterion: 'torch.nn.modules.loss' = nn.MSELoss(),
                 optimizer: 'torch.optim' = optim.Adam,
                 normalize_data: bool = True,
                 random_state: int = 2020,
                 verbose: bool = False,
                 weight_decay: float = 0.0,
                 patience: int = None,
                 val_fraction: float = 0.0,
                 use_scheduler: bool = False,
                 **kwargs):
        super().__init__(name=name,
                         window_size=window_size,
                         forecast_horizon=forecast_horizon,
                         target_name=target_name,
                         input_size=input_size,
                         hidden_size=hidden_size,
                         num_layers=num_layers,
                         num_classes=num_classes,
                         output_size=output_size,
                         cell_type=cell_type,
                         batch_size=batch_size,
                         n_epochs=n_epochs,
                         learning_rate=learning_rate,
                         criterion=criterion,
                         optimizer=optimizer,
                         normalize_data=normalize_data,
                         random_state=random_state,
                         verbose=verbose,
                         weight_decay=weight_decay,
                         patience=patience,
                         val_fraction=val_fraction,
                         use_scheduler=use_scheduler,
                         **kwargs)


class LSTMModel(RNNModel):
    """A Long Short-Term Memory (LSTM) forecasting model compatible with otbench."""

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
                 cell_type: str = 'lstm',
                 batch_size: int = 32,
                 n_epochs: int = 100,
                 learning_rate: float = 0.001,
                 criterion: 'torch.nn.modules.loss' = nn.MSELoss(),
                 optimizer: 'torch.optim' = optim.Adam,
                 normalize_data: bool = True,
                 random_state: int = 2020,
                 verbose: bool = False,
                 weight_decay: float = 0.0,
                 patience: int = None,
                 val_fraction: float = 0.0,
                 use_scheduler: bool = False,
                 **kwargs):
        super().__init__(name=name,
                         window_size=window_size,
                         forecast_horizon=forecast_horizon,
                         target_name=target_name,
                         input_size=input_size,
                         hidden_size=hidden_size,
                         num_layers=num_layers,
                         num_classes=num_classes,
                         output_size=output_size,
                         cell_type=cell_type,
                         batch_size=batch_size,
                         n_epochs=n_epochs,
                         learning_rate=learning_rate,
                         criterion=criterion,
                         optimizer=optimizer,
                         normalize_data=normalize_data,
                         random_state=random_state,
                         verbose=verbose,
                         weight_decay=weight_decay,
                         patience=patience,
                         val_fraction=val_fraction,
                         use_scheduler=use_scheduler,
                         **kwargs)
