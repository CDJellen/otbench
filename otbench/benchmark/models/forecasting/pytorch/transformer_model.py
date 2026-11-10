import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from otbench.benchmark.models.forecasting.pytorch.base_pytorch_model import BasePyTorchForecastingModel


class PositionalEncoding(nn.Module):
    """Standard Sinusoidal Positional Encoding."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        # Batch first: (1, max_len, d_model)
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class Transformer(nn.Module):
    """Time-Series Transformer Encoder."""

    def __init__(self,
                 input_size: int,
                 d_model: int,
                 nhead: int,
                 num_layers: int,
                 output_size: int,
                 dropout: float = 0.1):
        super(Transformer, self).__init__()

        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        encoder_layers = nn.TransformerEncoderLayer(d_model,
                                                    nhead,
                                                    dim_feedforward=d_model * 4,
                                                    dropout=dropout,
                                                    batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Linear(d_model, output_size)
        self.d_model = d_model

    def forward(self, x: torch.Tensor):
        x = self.input_projection(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        output = self.transformer_encoder(x)
        last_step_output = output[:, -1, :]
        prediction = self.decoder(last_step_output)
        return prediction


class TransformerModel(BasePyTorchForecastingModel):
    """A Transformer-based forecasting model compatible with otbench."""

    def __init__(self,
                 name: str,
                 window_size: int,
                 forecast_horizon: int,
                 target_name: str,
                 input_size: int,
                 d_model: int = 128,
                 nhead: int = 4,
                 num_layers: int = 2,
                 output_size: int = None,
                 dropout: float = 0.1,
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
        self.output_size = output_size if output_size is not None else 1

        model = Transformer(input_size=input_size,
                            d_model=d_model,
                            nhead=nhead,
                            num_layers=num_layers,
                            output_size=self.output_size,
                            dropout=dropout)

        self.set_model(model=model, normalize_data=normalize_data, set_optimizer_callable_params=True)

    def _train(self, X: 'pd.DataFrame', y: 'pd.DataFrame'):
        self._run_training_loop(X=X, y=y, input_size=self.input_size)

    def _predict(self, X: 'pd.DataFrame'):
        """Generate predictions from the TransformerModel."""
        return self._run_prediction(X=X, output_dim=self.output_size)


class PhysicsTransformerModel(TransformerModel):
    """A Physics-Informed Transformer forecasting model trained with AOPhysicsLoss."""

    def __init__(self,
                 name: str,
                 window_size: int,
                 forecast_horizon: int,
                 target_name: str,
                 input_size: int,
                 d_model: int = 128,
                 nhead: int = 4,
                 num_layers: int = 2,
                 output_size: int = None,
                 dropout: float = 0.1,
                 batch_size: int = 32,
                 n_epochs: int = 100,
                 learning_rate: float = 0.001,
                 criterion: 'torch.nn.modules.loss' = None,
                 optimizer: 'torch.optim' = optim.Adam,
                 normalize_data: bool = True,
                 random_state: int = 2020,
                 verbose: bool = False,
                 weight_decay: float = 0.0,
                 patience: int = None,
                 val_fraction: float = 0.0,
                 use_scheduler: bool = False,
                 use_physics_loss: bool = True,
                 lambda_r0: float = 0.15,
                 lambda_theta0: float = 0.15,
                 jensen_correction: bool = True,
                 **kwargs):
        super().__init__(name=name,
                         window_size=window_size,
                         forecast_horizon=forecast_horizon,
                         target_name=target_name,
                         input_size=input_size,
                         d_model=d_model,
                         nhead=nhead,
                         num_layers=num_layers,
                         output_size=output_size,
                         dropout=dropout,
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
                         use_physics_loss=use_physics_loss,
                         lambda_r0=lambda_r0,
                         lambda_theta0=lambda_theta0,
                         jensen_correction=jensen_correction,
                         **kwargs)
