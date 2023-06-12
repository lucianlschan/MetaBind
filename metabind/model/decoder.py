from collections import Counter
import torch
from torch import nn
from torch.distributions import categorical, gamma, bernoulli
from .mlp import MLP
from ..util.xcov import cross_covariance_aggregate


class ClusterModel(nn.Module):
    """
    Cluster Model
    """
    def __init__(self,
                input_dimension: int,
                n_clusters: int,
                decoder_hidden_dimension: int,
                dropout: float, 
                epsilon: float):
        super(ClusterModel, self).__init__()

        assert n_clusters > 1
        self.kdim = n_clusters
        if n_clusters == 2:
            self.aggregate = MLP(input_dimension, [decoder_hidden_dimension], 1 + 2, nn.Tanh(), dropout, batch_normalization=False)
        else:
            self.aggregate = MLP(input_dimension, [decoder_hidden_dimension], n_clusters + 2, nn.Tanh(), dropout, batch_normalization=False)

        # Activation functions
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()
        self.epsilon = epsilon

    def forward(self, index, x, y, anchor_x, anchor_y):
        """
        index:list
        x: tensor
        y: tensor
        : tensor Additional Information
        """
        sizelist = list(Counter(index).values())
        x_relative = x - torch.repeat_interleave(anchor_x, torch.LongTensor(sizelist), dim=0)
        y_relative = y - torch.repeat_interleave(anchor_y, torch.LongTensor(sizelist), dim=0)

        representation, _ = cross_covariance_aggregate(x_relative, y_relative, index)
        #xtxinv = torch.concat([1/x_.var(dim=0, keepdim=True) for i, x_ in enumerate(torch.split(x_relative, sizelist))])
        xtxinv = torch.concat([1/x_.var().reshape([1,1]) for i, x_ in enumerate(torch.split(x_relative, sizelist))])
        representation = torch.concat([xtxinv*representation], dim=1)

        if self.kdim == 2:
            alpha, beta, mixture = torch.split( self.aggregate(representation), [1, 1, 1], dim=1 )
            alpha, beta = self.softplus(alpha)*(1 - self.epsilon) + self.epsilon,  self.softplus(beta)*(1 - self.epsilon) + self.epsilon
            precision_dist = gamma.Gamma(alpha, beta)
            precision_mean = alpha/beta
            scale = torch.sqrt(1/precision_mean)
            prob = self.sigmoid(mixture)
            mixture = torch.concat([prob, 1 - prob], dim=1)
            mixture_dist = bernoulli.Bernoulli(prob)
        else:
            alpha, beta, mixture = torch.split( self.aggregate(representation), [1, 1, self.kdim], dim=1 )
            alpha, beta = self.softplus(alpha)*(1 - self.epsilon) + self.epsilon, self.softplus(beta)*(1 - self.epsilon) + self.epsilon
            precision_dist = gamma.Gamma(alpha, beta)
            precision_mean = alpha/beta
            scale = torch.sqrt(1/precision_mean)
            mixture = self.softmax(mixture)
            mixture_dist = categorical.Categorical(mixture)
        return representation, mixture, mixture_dist, scale, precision_dist

