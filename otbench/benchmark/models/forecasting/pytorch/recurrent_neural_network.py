import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from otbench.benchmark.models.forecasting.pytorch.base_pytorch_model import BasePyTorchForecastingModel


class RNN(nn.Module):
    """A configurable recurrent module supporting RNN, LSTM, and GRU cells.

    The cell_type parameter selects the recurrence mechanism:
      - 'rnn':  Vanilla Elman RNN (no gating — susceptible to vanishing gradients)
      - 'lstm': Long Short-Term Memory (forget/input/output gates)
      - 'gru':  Gated Recurrent Unit (update/reset gates — fewer params than LSTM)

    All three share the same hidden→output projection (fc layer) and are
    interchangeable from the perspective of the training loop.
    """

    _CELL_TYPES = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}

    def __init__(self, input_size, hidden_size, num_layers, num_classes, cell_type='rnn'):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell_type = cell_type

        if input_size <= 0:
            raise AssertionError("input size must be greater than 0")
        if cell_type not in self._CELL_TYPES:
            raise ValueError(f"Unsupported cell_type '{cell_type}'. Must be one of {list(self._CELL_TYPES.keys())}")

        cell_cls = self._CELL_TYPES[cell_type]
        self.rnn = cell_cls(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        if self.cell_type == 'lstm':
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
            out, _ = self.rnn(x, (h0, c0))
        else:
            out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out


class RNNModel(BasePyTorchForecastingModel):
    """A configurable PyTorch recurrent model supporting RNN, LSTM, and GRU cells.

    The ``cell_type`` parameter selects the recurrence architecture:
      - ``'rnn'``:  Vanilla Elman RNN
      - ``'lstm'``: Long Short-Term Memory
      - ``'gru'``:  Gated Recurrent Unit

    All other parameters (hidden_size, num_layers, training protocol) are
    shared across cell types, enabling controlled architectural comparison.
    """

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

        # create and set the model
        model = RNN(input_size, hidden_size, num_layers, self.num_classes, cell_type=cell_type)
        self.set_model(model=model, normalize_data=normalize_data,
                       set_optimizer_callable_params=True)  # apply model params to optimizer

    def _train(self, X: 'pd.DataFrame', y: 'pd.DataFrame'):
        self._run_training_loop(X=X, y=y, input_size=self.input_size)

    def _predict(self, X: 'pd.DataFrame'):
        """Generate predictions from the RNNModel."""
        return self._run_prediction(X=X, output_dim=self.num_classes)