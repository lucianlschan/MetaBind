import torch
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter_sum, scatter_softmax
import dgl
from .conv import SequenceConv, GraphConvLayer
from .mlp import MLP

class SequenceNN(nn.Module):
    def __init__(self,
                in_channels=[1, 1],
                out_channels=[1, 1],
                kernelsize=[(3,31),(16,1)],
                stride = [1, 4],
                pooling = ["Max","Max"],
                layernorm = [True, True],
                input_dimension = (1500,31),
                bias=False):
        super(SequenceNN, self).__init__()
        self.nlayers = len(kernelsize)
        self.layernorm = layernorm
        self.convlayers = nn.ModuleList()
        self.normlayers = []
        indim = input_dimension
        for idx in range(self.nlayers):
            self.convlayers.append(
                            SequenceConv(in_channels = in_channels[idx],
                                       out_channels = out_channels[idx],
                                        kernelsize = kernelsize[idx],
                                        stride = stride[idx],
                                        normalize = layernorm[idx],
                                        input_dimension = indim,
                                        bias = bias)
                            )

            indim = (int((indim[0] - kernelsize[idx][0] + 1)/stride[idx]), kernelsize[idx][1])
            if self.layernorm[idx]:
                self.normlayers.append(nn.LayerNorm(indim))

    def forward(self, x):
        for idx in range(self.nlayers):
            x = self.convlayers[idx](x)
            if self.layernorm[idx]:
                x = self.normlayers[idx](x)
        return x

class SequenceEncoder(nn.Module):
    def __init__(self,
                in_channels = [1, 1],
                out_channels = [1, 1],
                kernelsize=[(3,33), (16,1)],
                stride = [1,4],
                pooling = ["Max","Max"],
                layernorm = [False, False],
                input_dimension = (1500, 33),
                bias = False):
        super(SequenceEncoder, self).__init__()
        self.seqnn = SequenceNN(in_channels= in_channels,
                                out_channels= out_channels,
                                kernelsize= kernelsize,
                                stride = stride,
                                pooling = pooling,
                                layernorm = layernorm,
                                input_dimension = input_dimension,
                                bias = False)
    def forward(self, X):
        return self.seqnn(X).flatten(1)



class GCN(nn.Module):
    def __init__(self,
                in_features,
                hidden_features=[64,64],
                activation=[F.relu, F.relu],
                gnn_norm=["none", "none"],
                residual=[True, True],
                batchnorm=[True, True],
                dropout=[0., 0.],
                bias = False):
        super(GCN, self).__init__()
        nlayers = len(hidden_features)
        assert len(hidden_features) == nlayers
        if len(gnn_norm) != nlayers: gnn_norm = ["none"]*nlayers
        if len(batchnorm) != nlayers: batchnorm = [True]*nlayers
        if len(activation) != nlayers: activation = [F.relu]*nlayers
        if len(dropout) != nlayers: dropout = [0.]*nlayers
        if len(residual) != nlayers: residual = [True]*nlayers
        self.hidden_features = hidden_features
        self.layers = nn.ModuleList()
        for i in range(nlayers):
            self.layers.append(GraphConvLayer(in_features = in_features,
                                        out_features = hidden_features[i],
                                        gnn_norm = gnn_norm[i],
                                        activation = activation[i],
                                        residual = residual[i],
                                        batchnorm = batchnorm[i],
                                        dropout = dropout[i],
                                        bias = bias))
            in_features = hidden_features[i]

    def reset_parameters(self):
        for layers in self.layers:
            layers.reset_parameters()

    def forward(self, g, x):
        for layer in self.layers:
            x = layer(g, x)
        return x


class GraphPooling(nn.Module):
    def __init__(self, method):
        super(GraphPooling, self).__init__()
        assert len(method) > 0
        self.method = method

    def forward(self, G, G_feat):
        tensor = []
        with G.local_scope():
            G.ndata["z"] = G_feat
            for pool in self.method:
                if "Max" == pool:
                    G_max = dgl.max_nodes(G, "z")
                    tensor.append(G_max)
                elif "Sum" in self.method:
                    G_sum = dgl.sum_nodes(G, "z")
                    tensor.append(G_sum)
                elif "Mean" in self.method:
                    G_mean = dgl.mean_nodes(G, "z")
                    tensor.append(G_mean)
        output = torch.concat(tensor, dim=1)
        return output



class LigandEncoder(nn.Module):
    def __init__(self,
                in_features = 34,
                hidden_features = [128,128],
                activation=[F.relu, F.relu],
                gnn_norm=["none", "none"],
                residual=[True, True],
                batchnorm=[False, False],
                bias = False,
                graph_pooling = ["Sum"]):
        super(LigandEncoder, self).__init__()

        gcn_input_dim = hidden_features[0]
        self.ligandmap = nn.Linear(in_features = in_features,
                                   out_features=gcn_input_dim, bias=bias)
        self.ligandgcn = GCN(in_features = gcn_input_dim,
                             hidden_features = hidden_features,
                             activation = activation,
                             batchnorm = batchnorm,
                             residual = residual,
                             bias = bias)
        self.graphpooling = graph_pooling
        if graph_pooling is not None:
            self.gmap = GraphPooling(method=graph_pooling)


    def forward(self, G):
        with G.local_scope():
            G.ndata["zh"] = self.ligandmap(G.ndata["h"])
            G.ndata["z"] = self.ligandgcn(G, G.ndata["zh"])
            G_feat = G.ndata["z"]
            if self.graphpooling is not None:
                return self.gmap(G, G_feat)
            else:
                return G, G_feat

