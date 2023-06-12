import math
import pickle
import numpy as np
import torch
from torch import nn
import dgl
from metabind.model.encoder import LigandEncoder, SequenceEncoder
from metabind.model.mlp import MLP



class AggregatedModel(nn.Module):
    def __init__(self,
                ligand_in_dimension = 34,
                ligand_latent_dimension = 128,
                ligand_graph_pooling = ["Sum", "Max"],
                sequence_in_dimension = (1500, 32),
                sequence_in_channels = [1, 1],
                sequence_out_channels = [1, 1],
                sequence_kernel_size = [(10, 32), (10, 1)],
                sequence_stride = [3, 3],
                decoder_in_dimension = "Infer",
                decoder_hidden_dimension = [128],
                decoder_out_dimension = 1):
        super(AggregatedModel, self).__init__()

        ligand_out_dimension = ligand_latent_dimension*len(ligand_graph_pooling)
        sequence_init_dim = sequence_in_dimension

        sequence_rep_dim = sequence_init_dim[0]
        for idx, sks in enumerate(sequence_kernel_size):
            sequence_rep_dim = math.ceil((sequence_rep_dim - sequence_kernel_size[idx][0] + 1)/sequence_stride[idx])
        sequence_out_rep_dim = sequence_rep_dim

        protein_ligand_concat_dimension = ligand_out_dimension +  sequence_out_rep_dim
        decoder_in_dimension = protein_ligand_concat_dimension if decoder_in_dimension == "Infer" else decoder_in_dimension

        self.seqnn = SequenceEncoder(kernelsize = sequence_kernel_size, stride= sequence_stride)
        self.lignn = LigandEncoder(in_features = 34,
                                   hidden_features = [ligand_latent_dimension, ligand_latent_dimension],
                                   residual = ["True", "True"],
                                   batchnorm = ["True", "True"],
                                   graph_pooling = ligand_graph_pooling)

        self.pl = nn.Linear(decoder_in_dimension, decoder_in_dimension)
        self.decoder = MLP(decoder_in_dimension,
                           decoder_hidden_dimension,
                           decoder_out_dimension,
                           nn.ELU(),
                           dropout = 0.4,
                           batch_normalization=False)

    
    def forward(self, protein, ligand):
        batch_ligand = dgl.batch(ligand)
        batch_ligand_feature = self.lignn(batch_ligand)
        sequence_feature = self.seqnn(protein)
        x = torch.concat([sequence_feature, batch_ligand_feature], dim=1)
        return self.decoder(self.pl(x))


    

