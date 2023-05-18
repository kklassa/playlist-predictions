import torch
from torch import nn


class NNPredictor(nn.Module):
    def __init__(self, num_features, num_hidden) -> None:
        super(NNPredictor, self).__init__()
        self.input_layer = nn.Linear(num_features, num_hidden)
        self.hidden_layer_1 = nn.Linear(num_hidden, num_hidden)
        self.hidden_layer_2 = nn.Linear(num_hidden, num_hidden)
        self.output_layer = nn.Linear(num_hidden, 1)

        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.input_layer(x))
        x = self.activation(self.hidden_layer_1(x))
        x = self.activation(self.hidden_layer_2(x))
        return self.output_layer(x)
