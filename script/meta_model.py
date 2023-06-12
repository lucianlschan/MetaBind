import math
import pickle
from collections import Counter
import numpy as np
import torch
from torch import nn
from torch.distributions import normal
from torch.distributions import kl_divergence
from torch_scatter import scatter_mean
import dgl
from metabind.model.decoder import ClusterModel
from metabind.model.mlp import MLP
from metabind.model.encoder import LigandEncoder, SequenceEncoder

class MetaBind(nn.Module):
    def __init__(self,
                ligand_input_dimension = 34,
                ligand_dimension=128,
                ligand_graph_pooling= ["Sum","Max"],
                sequence_input_dimension = (1500, 32),
                sequence_in_channels = [1, 1],
                sequence_out_channels = [1, 1],
                sequence_kernel_size = [(10,32), (10, 1)],
                sequence_stride = [3, 3],
                y_output_dimension = 1,
                PL_dimension = 256,
                n_clusters = 4,
                decoder_hidden_dimension = 128,
                epsilon = 0.01):

        """
                mixture_dropout = 0.3,
                mixture_hidden_dimension = 128,
                mixture_activation_function = nn.Tanh(),
                decoder_dropout = 0.3,
                decoder_hidden_dimension = 128,
                decoder_activation_function = nn.Tanh(),
                learnable_scale = True,
                data_scale_epsilon = 0.01,
                decoder_scale_epsilon = 0.01):
        """
        super(MetaBind, self).__init__()

        # Model Input Dimension
        ligand_output_dim = ligand_dimension*len(ligand_graph_pooling)
        sequence_init_dim = sequence_input_dimension
        
        sequence_rep_dim = sequence_init_dim[0]
        for idx, sks in enumerate(sequence_kernel_size):
            sequence_rep_dim = math.ceil((sequence_rep_dim - sequence_kernel_size[idx][0] + 1)/sequence_stride[idx])
        sequence_output_dim = sequence_rep_dim
        
        y_dimension = 1

        # Models
        self.seqnn = SequenceEncoder(in_channels = sequence_in_channels,
                                            out_channels = sequence_out_channels,
                                            kernelsize = sequence_kernel_size, 
                                            stride = sequence_stride) 

        self.lignn = LigandEncoder(in_features=ligand_input_dimension, 
                                          hidden_features=[ligand_dimension, ligand_dimension],
                                          residual = ["True","True"],
                                          batchnorm = ["True", "True"], 
                                          graph_pooling=ligand_graph_pooling)

        self.pl = nn.Linear(sequence_output_dim + ligand_output_dim, PL_dimension)

        self.cluster = ClusterModel(input_dimension = PL_dimension,
                                     n_clusters = n_clusters,
                                     decoder_hidden_dimension = decoder_hidden_dimension,
                                     dropout = 0.3,
                                     epsilon = 0.01)

        self.decoder = MLP(in_dimension = PL_dimension,
                           hidden_dimension = [decoder_hidden_dimension],
                           out_dimension = 2*n_clusters,
                           activation = nn.Tanh(),
                           dropout = 0.3,
                           batch_normalization = False)
        
        self.softplus = nn.Softplus()       
        self.nclusters = n_clusters
        self.ydim = y_dimension
        self.epsilon = epsilon

    def featurize_sequence(self, sequence):
        sequence_feature = self.seqnn(sequence)
        return sequence_feature

    def featurize_ligand(self, ligand):
        ligand_graph_batch = dgl.batch(ligand)
        ligand_graph_feature = self.lignn(ligand_graph_batch)
        return ligand_graph_feature
    
    def concatenate_features(self, sequence, ligand):
        return self.pl(torch.concat([sequence, ligand], dim=1))
    
    def forward(self, context_index, context_x, context_y, target_index, target_x, target_y):
        target_sizelist = list(Counter(target_index).values())
        context_sizelist = list(Counter(context_index).values())
        
        context_x_anchor = scatter_mean(context_x, torch.LongTensor(context_index), dim=0)
        context_y_anchor = scatter_mean(context_y, torch.LongTensor(context_index), dim=0)
        context_rep,  context_mixture_prob, context_mixture_dist, context_scale, context_precision_dist = self.cluster(context_index, context_x, context_y, context_x_anchor, context_y_anchor)

        if target_y is not None:

            target_x_anchor = scatter_mean(target_x, torch.LongTensor(target_index), dim=0)
            target_y_anchor = scatter_mean(target_y, torch.LongTensor(target_index), dim=0)
            target_rep, target_mixture_prob, target_mixture_dist, target_scale, target_precision_dist = self.cluster(target_index, target_x, target_y, target_x_anchor, target_y_anchor)
            target_x_relative = target_x - target_x_anchor[target_index]

            #target_prediction, target_instance_dist = torch.split( self.decoder( target_x_relative ), [self.nclusters, self.nclusters], dim=1)
            target_prediction = self.decoder(target_x_relative)
            target_mean, target_log_sigma = torch.split(target_prediction, [self.nclusters, self.nclusters], dim=1)
            target_sigma = self.softplus(target_log_sigma)*(1 - self.epsilon) + self.epsilon 
            target_scaled_mean = (target_scale[target_index])*target_mean

            # Prediction
            target_y_anchor_full = torch.repeat_interleave(target_y_anchor, torch.LongTensor(target_sizelist), dim=0)
            y_pred = (target_mixture_prob[target_index]*target_scaled_mean).sum(dim=1, keepdim=True) + target_y_anchor_full
            y_sigma = (target_scale*target_mixture_prob)[target_index]*target_sigma
            y_sigma_sum = torch.sqrt((y_sigma**2).sum(dim=1))
            

            mixture_kl = kl_divergence(target_mixture_dist, context_mixture_dist)
            precision_kl = kl_divergence(target_precision_dist, context_precision_dist)
            dist_kl = mixture_kl + precision_kl 
            return y_pred, y_sigma_sum, target_mixture_prob, target_mixture_dist, target_rep, dist_kl, target_index, target_scale

        else:
            context_x_anchor_full = torch.repeat_interleave( context_x_anchor, torch.LongTensor(target_sizelist), dim=0 )
            target_x_relative = target_x - context_x_anchor_full
            
            target_prediction = self.decoder(target_x_relative)
            target_mean, target_log_sigma = torch.split( target_prediction , [self.nclusters, self.nclusters], dim=1)
            target_sigma = self.softplus(target_log_sigma)*(1 - self.epsilon) + self.epsilon
            target_scaled_mean = (context_scale[target_index])*target_mean
            target_y_anchor_full = torch.repeat_interleave(context_y_anchor, torch.LongTensor(target_sizelist), dim=0)
        
            # Prediction
            y_pred = (context_mixture_prob[target_index]*target_scaled_mean).sum(dim=1, keepdim=True) + target_y_anchor_full
            #y_sigma = (target_scale*target_mixture_prob)[target_index]*target_sigma
            #y_sigma_sum = torch.sqrt((y_sigma**2).sum(dim=1))
            y_sigma = (context_scale*context_mixture_prob)[target_index]*target_sigma
            y_sigma_sum = torch.sqrt((y_sigma**2).sum(dim=1))

            return y_pred, y_sigma_sum, context_mixture_prob,  context_mixture_dist, context_rep, target_index, context_scale
    """
    def forward(self, context_index, context_sequence, context_ligand, context_y, target_index, target_sequence, target_ligand, target_y):
        target_sizelist = list(Counter(target_index).values())
        context_sizelist = list(Counter(context_index).values())

        context_in_x = self.pl(torch.concat([context_sequence, context_ligand], dim=1))
        context_in_x_anchor = scatter_mean(context_in_x, torch.LongTensor(context_index), dim=0)
        context_y_anchor = scatter_mean(context_y, torch.LongTensor(context_index), dim=0)
        context_rep,  context_mixture_prob, context_mixture_dist, context_scale, context_precision_dist = self.model(context_index, context_in_x, context_y, context_in_x_anchor, context_y_anchor)
        
        if target_y is not None:

            target_in_x = self.pl(torch.concat([target_sequence, target_ligand], dim=1))
            target_in_x_anchor = scatter_mean(target_in_x, torch.LongTensor(target_index), dim=0)
            target_y_anchor = scatter_mean(target_y, torch.LongTensor(target_index), dim=0)
            target_rep, target_mixture_prob, target_mixture_dist, target_scale, target_precision_dist = self.model(target_index, target_in_x, target_y, target_in_x_anchor, target_y_anchor)
            target_in_x_relative = target_in_x - target_in_x_anchor[target_index]

            target_prediction, target_instance_dist = self.model.decoder( target_in_x_relative )
            target_instance, target_instance_sigma = torch.split( target_prediction , [self.kdim, self.kdim], dim=1)
            target_scaled_instance = (target_scale[target_index])*target_instance 
        
            # Prediction
            target_y_anchor_full = torch.repeat_interleave(target_y_anchor, torch.LongTensor(target_sizelist), dim=0)
            y_pred = (target_mixture_prob[target_index]*target_scaled_instance).sum(dim=1, keepdim=True) + target_y_anchor_full
            y_sigma = (target_scale*target_mixture_prob)[target_index]*target_instance_sigma

            mixture_kl = kl_divergence(target_mixture_dist, context_mixture_dist)
            precision_kl = kl_divergence(target_precision_dist, context_precision_dist)
            dist_kl = mixture_kl + precision_kl 
            return y_pred, y_sigma, target_mixture_prob, target_mixture_dist, target_rep, dist_kl, target_index, target_scale

        else:
            target_in_x = self.pl(torch.concat([target_sequence, target_ligand], dim=1))
            context_in_x_anchor_full = torch.repeat_interleave( context_in_x_anchor, torch.LongTensor(target_sizelist), dim=0 )
            target_in_x_relative = target_in_x - context_in_x_anchor_full
            
            target_prediction, target_instance_dist = self.model.decoder( target_in_x_relative )
            target_instance, target_instance_sigma = torch.split( target_prediction , [self.kdim, self.kdim], dim=1)
            target_scaled_instance = (context_scale[target_index])*target_instance 
            target_y_anchor_full = torch.repeat_interleave(context_y_anchor, torch.LongTensor(target_sizelist), dim=0)
        
            # Prediction
            y_pred = (context_mixture_prob[target_index]*target_scaled_instance).sum(dim=1, keepdim=True) + target_y_anchor_full
            y_sigma = (context_scale*context_mixture_prob)[target_index]*target_instance_sigma

            return y_pred, y_sigma, context_mixture_prob,  context_mixture_dist, context_rep, target_index, context_scale
    """
