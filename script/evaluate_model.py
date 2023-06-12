import json
import random
import logging
import os
import argparse
import math
import pickle
import torch
from torch import nn
from torch import optim
from torch.distributions.kl import kl_divergence
from aggregated_model import AggregatedModel
from meta_model import MetaBind

sequence_library = pickle.load(open("../data/sequence/sequence_feature_dictionary.pkl","rb"))

def existingFile(filename):
    if not os.path.exists(filename):
        raise argparse.ArgumentTypeError("{0} does not exists".format(filename))
    return filename

def feature_loader(ik, seqid, val, shift=-2.21):
    ligand = pickle.load(open("../data/molecule/{}.pkl".format(ik),"rb"))
    sequence = sequence_library.get(seqid).expand(1, 1, 1500, 32)
    lval = -math.log10(val) - shift
    return sequence, ligand, lval

class aggregated_dataloader:
    def __init__(self, data):
        self.data = data
    
    def get_batch(self, index_list):
        batch = [feature_loader(*self.data[i]) for i in index_list]
        batch_sequence, batch_ligand, batch_lval = list(zip(*batch))
        return batch_sequence, batch_ligand, batch_lval

class meta_dataloader:
    def __init__(self, data, sample_size="auto"):
        self.data = data
        self.samplesize = sample_size        

    def get_batch(self, key_list):
        batch_sequence, batch_ligand, batch_lval = [], [], []
        size = 0
        batch_context_index, batch_target_index = [], []
        batch_context_group_index, batch_target_group_index = [], []
        for idx, key in enumerate(key_list):
            assay = self.data[key]
            assay_size = len(assay)
            assay_batch = [feature_loader(*i) for i in assay]
            assay_sequence, assay_ligand, assay_lval = list(zip(*assay_batch))
            batch_sequence += assay_sequence
            batch_ligand += assay_ligand
            batch_lval += assay_lval
            if self.samplesize == "auto":
                sample_size = random.choice(range(int(assay_size/2),assay_size))
            else:
                sample_size = self.samplesize
            context_index, target_index = _sample_context(range(assay_size), sample_size = sample_size, shift = size)
            batch_context_index += context_index
            batch_target_index += target_index
            batch_context_group_index += [idx]*len(context_index)
            batch_target_group_index += [idx]*len(target_index)
            size += assay_size
        return batch_sequence, batch_ligand, batch_lval, batch_context_index, batch_target_index, batch_context_group_index, batch_target_group_index

def initParser():
    parser = argparse.ArgumentParser(description = "Aggregated/Meta Model Training")
    parser.add_argument("--model", type=str, help="aggregated or meta")
    parser.add_argument("--split", type=str, help="chronological or paired")
    parser.add_argument("--datatype", type=str, help="All/Biochemical/Cell")
    parser.add_argument("--parameter", type=str, help="Traind model parameter files")
    parser.add_argument("--ncluster", type=int, help="Number of clusters, only used in meta-model", default=4)
    return parser

def _sample_context(bag, sample_size, shift=0, replacement=True):
    context, target = [], []
    bagsize = len(bag)
    context_index = random.sample(range(bagsize), sample_size)
    if replacement:
        target_index = list(range(bagsize))
    else:
        target_index = list(set(range(bagsize)) - set(context_index))
    return [i + shift for i in context_index], [i + shift for i in target_index]

def main():
    if not os.path.exists("../evaluations"):
        os.mkdir("../evaluations")

    parser = initParser()
    args = parser.parse_args()
    parameterfile = existingFile(args.parameter)

    # Initialize model
    if args.split == "chronological":
        infile = json.load(open("../data/Chronological/chronological_test.json"))
        data = {}
        for key, item in infile.items():
            data[key] = [(i[0], int(i[1]), float(i[2])) for i in item]
        loader = meta_dataloader(data, sample_size=5)
        assay_keys = list(infile.keys())
    elif args.split == "paired":
        infile = json.load(open("../data/PairedSplit/paired_test.json"))
        data = {}
        for key, item in infile.items():
            i0, i1 = [(i[0], int(i[1]), float(i[2])) for i in item[0]], [(i[0], int(i[1]), float(i[2])) for i in item[1]]
            key0, key1 = "-".join(["A",key]), "-".join(["B",key])
            data[key0] = i0
            data[key1] = i1
        loader = meta_dataloader(data, sample_size=5)
        assay_keys = list(data.keys())
    elif args.split == "mutant":
        infile = json.load(open("../data/paired_mutant.json"))
        data = {}
        for key, item in infile.items():
            i0, i1 = [(i[0], int(i[1]), float(i[2])) for i in item[0]], [(i[0], int(i[1]), float(i[2])) for i in item[1]]
            key0, key1 = "-".join(["A",key]), "-".join(["B",key])
            data[key0] = i0
            data[key1] = i1
        loader = meta_dataloader(data, sample_size=10)
        assay_keys = list(data.keys())
    else:
        raise NotImplementedError           


    # Loss Function
    gaussnll = nn.GaussianNLLLoss(reduction="none")
    mseloss = nn.MSELoss(reduction="none")

    model_prediction = {}
    if args.model == "aggregated":
        model = AggregatedModel()
        parameters = torch.load(parameterfile)
        model.load_state_dict(parameters["Model"])
        for key in assay_keys:
            # Standard Prediction
            model.eval()
            batch_sequence, batch_ligand, batch_value, _, _, _, _ = loader.get_batch([key])           
            batch_sequence = torch.concat(batch_sequence)
            standard_prediction = model(batch_sequence, batch_ligand)
            
            # Local Prediction (one step gradient update)
            duplicated_model = AggregatedModel()
            duplicated_model.load_state_dict(parameters["Model"])
            # Optimizer
            optimizer = optim.Adam(duplicated_model.parameters(), lr=1e-5)
         
            duplicated_model.train(True) 
            assay_size = len(data[key])  
            indices = random.sample(range(assay_size), 5)
            _prediction = duplicated_model(batch_sequence, batch_ligand)[indices]
            _batch_value = torch.tensor(batch_value).reshape([len(batch_value), 1])[indices]
            loss = mseloss(_prediction, _batch_value).mean()           
            # Local Update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            duplicated_model.eval()
            local_prediction = duplicated_model(batch_sequence, batch_ligand)
            model_prediction[key] = [data[key], standard_prediction, local_prediction]
            del duplicated_model, optimizer
    else: # Meta
        model = MetaBind()
        parameters = torch.load(parameterfile)
        model.load_state_dict(parameters["Model"])
        for key in assay_keys:
            model.eval()
            batch_sequence, batch_ligand, batch_lval, batch_context_index, batch_target_index, batch_context_group_index, batch_target_group_index = loader.get_batch([key])
            batch_sequence = torch.concat(batch_sequence)
            batch_sequence_feature = model.featurize_sequence(batch_sequence)
            batch_ligand_feature = model.featurize_ligand(batch_ligand)
            batch_x_feature = model.concatenate_features(batch_sequence_feature, batch_ligand_feature)
            batch_y = torch.tensor(batch_lval).reshape([len(batch_lval), 1])
            context_x, context_y, target_x, target_y =  batch_x_feature[batch_context_index], batch_y[batch_context_index], batch_x_feature, batch_y
            prediction, sigma, mixture_prob, mixture_dist, mixture_rep,  _, _ = model(batch_context_group_index, context_x, context_y, batch_target_group_index, target_x, None)
            model_prediction[key] = [data[key], batch_context_index, mixture_prob, prediction]

    pickle.dump(model_prediction, open("../evaluations/{}_model_{}_evaluation.pkl".format(args.model, args.split), "wb"))
    
    
    


if __name__ == "__main__":
    main()




