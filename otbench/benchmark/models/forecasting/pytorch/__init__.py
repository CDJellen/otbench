from otbench.benchmark.models.forecasting.pytorch.base_pytorch_model import AOPhysicsLoss
from otbench.benchmark.models.forecasting.pytorch.recurrent_neural_network import (
    GRUModel,
    LSTMModel,
    RNNModel,
)
from otbench.benchmark.models.forecasting.pytorch.transformer_model import (
    PhysicsTransformerModel,
    TransformerModel,
)

__all__ = ["RNNModel", "TransformerModel"]
__all_extended__ = [
    "RNNModel",
    "GRUModel",
    "LSTMModel",
    "TransformerModel",
    "PhysicsTransformerModel",
]
