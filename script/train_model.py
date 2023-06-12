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

def feature_loader(ik, seqid, val, shift=-2.17):
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
                sample_size = random.choice(range(int(assay_size)/2, assay_size), self.samplesize)
            context_index, target_index = _sample_context(range(assay_size), sample_size = sample_size, shift = size)
            batch_context_index += context_index
            batch_target_index += target_index
            batch_context_group_index += [idx]*len(context_index)
            batch_target_group_index += [idx]*len(target_index)
            size += assay_size
        return batch_sequence, batch_ligand, batch_lval, batch_context_index, batch_target_index, batch_context_group_index, batch_target_group_index

def _sample_context(bag, sample_size, shift=0, replacement=True):
    context, target = [], []
    bagsize = len(bag)
    context_index = random.sample(range(bagsize), sample_size)
    if replacement:
        target_index = list(range(bagsize))
    else:
        target_index = list(set(range(bagsize)) - set(context_index))
    return [i + shift for i in context_index], [i + shift for i in target_index]

def initParser():
    parser = argparse.ArgumentParser(description = "Aggregated/Meta Model Training")
    parser.add_argument("--data", type=str, help="Data File")
    parser.add_argument("--model", type=str, help="aggregated or meta")
    parser.add_argument("--split", type=str, help="chronological or paired")
    parser.add_argument("--datatype", type=str, help="All/Biochemical/Cell")
    parser.add_argument("--nepoch", type=int, help="Number of epoch used in training")
    parser.add_argument("--batchsize", type=int, help="Number of assays (meta-model) or number of observations (aggregated-model)")
    parser.add_argument("--checkpoint", type=int, help="Checkpoint", default=500)
    parser.add_argument("--warmstart", type=int, help="Warmstart", default=500)
    parser.add_argument("--ncluster", type=int, help="Number of clusters, only used in meta-model", default=4)
    return parser


def main():
    if not os.path.exists("../trained_models"):
        os.mkdir("../trained_models")
    if not os.path.exists("../log"):
        os.mkdir("../log")

    parser = initParser()
    args = parser.parse_args()
    data = existingFile(args.data)
    
    model_name = args.model
    split_method = args.split
    data_type = args.datatype

    # Initialize model
    if model_name == "aggregated":
        model = AggregatedModel()
        infile = open(args.data,"r").readlines()[1:]
        data = []
        for line in infile:
            ik, seqid, val = line.split("\n")[0].split(",")
            data.append((ik, int(seqid), float(val)))
        datakey = list(range(len(data)))
        train_size = int(len(data)*0.9)
        train_index = random.sample(datakey, train_size)
        validation_index = list(set(datakey) - set(train_index))
        train_key, validation_key = [datakey[i] for i in train_index], [datakey[i] for i in validation_index]
        loader = aggregated_dataloader(data)
    else: # Meta Model
        model = MetaBind()
        infile = json.load(open(args.data)) 
        data = {}
        for key, item in infile.items():
            data[key] = [(i[0], int(i[1]), float(i[2])) for i in item]
        datakey = list(data.keys())
        train_size = int(len(datakey)*0.9)
        train_key = random.sample(datakey, train_size)
        validation_key = list(set(datakey) - set(train_key))
        loader = meta_dataloader(data)


    # Loss Function
    gaussnll = nn.GaussianNLLLoss(reduction="none")
    mseloss = nn.MSELoss(reduction="none")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    # Logging
    iteration, cpt = 0, 0
    logging.basicConfig(filemode="w", level=logging.INFO, filename="../log/{}_{}_{}.log".format(model_name, split_method, data_type))
    logfile = logging.getLogger("Model")
    

    logging.info("Start Training")


    for epoch in range(args.nepoch):
        random.shuffle(train_key)
        for idx in range(0, train_size, args.batchsize):
            model.train(True)
            batch_train_key = train_key[idx:idx + args.batchsize]
            if args.model == "aggregated":
                batch_train_sequence, batch_train_ligand, batch_train_value = loader.get_batch(batch_train_key)
                batch_train_sequence = torch.concat(batch_train_sequence)
                train_prediction = model(batch_train_sequence ,batch_train_ligand)
                train_y = torch.tensor(batch_train_value).reshape([len(batch_train_value), 1])
                loss = mseloss(train_prediction, train_y).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else: # Meta Model
                batch_train_sequence, batch_train_ligand, batch_lval, batch_context_index, batch_target_index, batch_context_group_index, batch_target_group_index = loader.get_batch(batch_train_key)
                batch_train_sequence = torch.concat(batch_train_sequence)
                batch_sequence_feature = model.featurize_sequence(batch_train_sequence)
                batch_ligand_feature = model.featurize_ligand(batch_train_ligand)
                batch_x_feature = model.concatenate_features(batch_sequence_feature, batch_ligand_feature)

                batch_y = torch.tensor(batch_lval).reshape([len(batch_lval), 1])
                context_x, context_y, target_x, target_y =  batch_x_feature[batch_context_index], batch_y[batch_context_index], batch_x_feature, batch_y
                prediction, sigma, mixture_prob, mixture_dist, mixture_rep, kl, _, _ = model(batch_context_group_index, context_x, context_y, batch_target_group_index, target_x, target_y)
                var = sigma**2
                loss = gaussnll(prediction, batch_y, var) + 0.5*kl[batch_target_group_index]
                loss = loss.mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                #print(loss.item())

            iteration += 1


            model.train(False)
            if args.model == "aggregated":
                batch_validation_key = random.sample(validation_key, args.batchsize)
                batch_validation_sequence, batch_validation_ligand, batch_validation_value = loader.get_batch(batch_validation_key)
                batch_validation_sequence = torch.concat(batch_validation_sequence)
                validation_prediction = model(batch_validation_sequence, batch_validation_ligand)
                validation_y = torch.tensor(batch_validation_value).reshape([len(batch_validation_value), 1])
                validation_loss = mseloss(validation_prediction, validation_y).mean()            
                logging.info("Epoch {}: {} {}".format(epoch, loss.item(), validation_loss.item()))
            else:
                batch_validation_key = random.sample(validation_key, args.batchsize)
                batch_validation_sequence, batch_validation_ligand, batch_validation_lval, batch_context_index, batch_target_index, batch_context_group_index, batch_target_group_index = loader.get_batch(batch_validation_key)
                batch_validation_sequence = torch.concat(batch_validation_sequence)
                batch_validation_sequence_feature = model.featurize_sequence(batch_validation_sequence)
                batch_validation_ligand_feature = model.featurize_ligand(batch_validation_ligand)
                batch_validation_x_feature = model.concatenate_features(batch_validation_sequence_feature, batch_validation_ligand_feature)

                batch_validation_y = torch.tensor(batch_validation_lval).reshape([len(batch_validation_lval), 1])
                context_x, context_y, target_x, target_y =  batch_validation_x_feature[batch_context_index], batch_validation_y[batch_context_index], batch_validation_x_feature, batch_validation_y
                validation_prediction, validation_sigma, validation_mixture_prob, validation_mixture_dist, validation_mixture_rep, _, _ = model(batch_context_group_index, context_x, context_y, batch_target_group_index, target_x, None)
                validation_var = validation_sigma**2
                validation_gaussloss = gaussnll(validation_prediction, batch_validation_y, validation_var).mean()
                validation_kl = kl.mean()
                validation_mse = mseloss(validation_prediction, batch_validation_y).mean()            
                logging.info("Epoch {}: {} {} {} {}".format(epoch, loss.item(), validation_gaussloss.item(), validation_kl.item(), validation_mse.item()))

            if iteration > args.warmstart:
                if iteration%args.checkpoint == 0:
                    checkptdict = {"Model": model.state_dict()}
                    torch.save(checkptdict, "../trained_models/{}_{}_{}_{}.pth".format(model_name, split_method, data_type, cpt))
                    cpt += 1

            del loss

    logging.info("Completed")
    logging.getLogger("Training")
    

if __name__ == "__main__":
    main()

