import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self,
                in_dimension:int,
                hidden_dimension:list,
                out_dimension:int,
                activation = nn.ReLU(),
                dropout = float,
                batch_normalization=bool):

        super(MLP, self).__init__()
        assert len(hidden_dimension) > 0
        self.activation = activation
        self.layers = nn.ModuleList()
        self.layers.append(nn.Dropout(dropout))
        self.layers.append(nn.Linear(in_dimension, hidden_dimension[0]))
        self.layers.append(activation)
        if batch_normalization:
            self.layers.append(nn.BatchNorm1d(hidden_dimension[0]))
        for idx, dim in enumerate(hidden_dimension):
            if idx == len(hidden_dimension) - 1:
                self.layers.append(nn.Linear(hidden_dimension[idx], out_dimension))
            else:
                self.layers.append(nn.Linear(hidden_dimension[idx], hidden_dimension[idx+1]))
                self.layers.append(self.activation)
                if batch_normalization:
                    self.layers.append(nn.BatchNorm1d(hidden_dimension[idx+1]))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


