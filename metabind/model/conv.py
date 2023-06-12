from torch import nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv


class SequenceConv(nn.Module):
    def __init__(self,
                in_channels=int,
                out_channels=int,
                kernelsize=tuple,
                stride=int,
                activation=str,
                normalize=bool,
                bias=bool,
                input_dimension=tuple ):

        super(SequenceConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernelsize, stride=stride, bias=bias)
        self.activation = nn.ReLU()
        self.normalize = normalize
        if self.normalize:
            self.layernorm = nn.LayerNorm([input_dimension[0] - kernelsize[0] + 1, input_dimension[1] - kernelsize[1] + 1])

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        if self.normalize:
            self.layernorm(x)
        return x

class GraphConvLayer(nn.Module):
    def __init__(self,
                in_features,
                out_features,
                gnn_norm=None,
                residual=True,
                batchnorm=True,
                activation=F.relu,
                dropout = 0.,
                bias = False):

        super(GraphConvLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.conv = GraphConv(in_feats=in_features,
                              out_feats=out_features,
                              norm=gnn_norm,
                              activation=activation,
                              bias=bias)

        self.residual = residual
        if self.residual:
            self.res_connection = nn.Linear(in_features, out_features, bias=bias)

        self.bn = batchnorm
        if batchnorm:
            self.bn_layer = nn.BatchNorm1d(out_features)

    def reset_parameters(self):
        self.conv.reset_parameters()
        if self.residual:
            self.res_connection.reset_parameters()
        if self.bn:
            self.bn_layer.reset_parameters()

    def forward(self, g, X):
        gX = self.conv(g, X)
        if self.residual:
            res_feats = self.activation(self.res_connection(X))
            gX = gX + res_feats
        gX = self.dropout(gX)
        if self.bn:
            gX = self.bn_layer(gX)
        return gX


