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

    def __init__(
            self,
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
        self.output_size = output_size if output_size is not None else 1

        model = Transformer(input_size=input_size,
                            d_model=d_model,
                            nhead=nhead,
                            num_layers=num_layers,
                            output_size=self.output_size,
                            dropout=dropout)

        self.set_model(model=model, normalize_data=normalize_data, set_optimizer_callable_params=True)

    def _train(self, X: 'pd.DataFrame', y: 'pd.DataFrame'):
        # Guard against empty data
        if len(X) == 0:
            return

        # 1. Dimension Guard — matches the reshape logic in _set_dataloader_from_data:
        # temporal mode when columns divide evenly, flat mode otherwise.
        n_cols = len(X.columns)
        if self.window_size > 1 and n_cols % self.window_size == 0:
            n_features_in_data = n_cols // self.window_size
        else:
            n_features_in_data = n_cols  # flat/single-step mode
        if n_features_in_data != self.input_size:
            raise ValueError(f"Dimension Mismatch: Model initialized with input_size={self.input_size}, "
                             f"but training data has {n_features_in_data} features per timestep.")

        if self.verbose:
            print(f"Training Transformer on {n_features_in_data} features with d_model={self.model.d_model}...")

        # 2. Standard Training Loop
        self.set_training_data(X=X, y=y)
        torch.manual_seed(self.random_state)

        for i in range(self.n_epochs):
            total_loss = 0
            for _, (X_batch, y_batch) in enumerate(self.train_dataloader):
                self.optimizer.zero_grad()
                X_batch, y_batch = X_batch.to(self.device).float(), y_batch.to(self.device).float()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                total_loss += loss.item()

            if self.verbose and self.n_epochs >= 10 and (i % (self.n_epochs // 10) == 0):
                avg_loss = total_loss / len(self.train_dataloader)
                print(f"Epoch {i}: Avg Loss: {avg_loss:.6f}")

    def _predict(self, X: 'pd.DataFrame'):
        if len(X) == 0:
            return np.empty((0, self.output_size))

        if self.verbose:
            print(f"Generating predictions with Transformer...")

        self.set_validation_data(X=X, y=None)
        pred = []

        with torch.no_grad():
            self.model.eval()
            for _, (X_batch, _) in enumerate(self.val_dataloader):
                X_batch = X_batch.to(self.device).float()
                y_pred = self.model(X_batch)

                if self.normalize_data:
                    y_pred = y_pred.cpu()
                    y_pred = y_pred * self.y_std + self.y_mean

                y_pred = y_pred.cpu().numpy()

                # Handle Scalar vs Vector output format
                if self.output_size == 1:
                    pred.append(y_pred[0][0])
                else:
                    pred.append(y_pred[0])

        self.model.train()  # Reset to train mode
        return np.array(pred)