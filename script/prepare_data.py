import os
import json
import pickle
import numpy as np
from scipy.stats import gmean
from rdkit import Chem
import torch
import dgl
from metabind.featurizer import seqfeaturizer, molfeaturizer

"""
Preprare chronological (before and after 2016) and paired assay split data and pre-generated molecular graph and sequence feature

"""


def main():
    
    if not os.path.exists("../data/Chronological"):
        os.mkdir("../data/Chronological")
    chronological_train, chronological_test = {}, {}
    
        
    if not os.path.exists("../data/PairedSplit"):
        os.mkdir("../data/PairedSplit")
    unpaired_train, paired_test = {}, {}
    
    if not os.path.exists("../data/Aggregated"):   
        os.mkdir("../data/Aggregated")
    
    
    # Read Data
    data = open("../data/chembl_30.txt","r").readlines()
    paired_assay_list = open("../data/paired_assayid.txt","r").readlines()
    paired_assay_aid = set([line.split("\n")[0].split(",")[0] for line in paired_assay_list[1:]])
    assay_group = {}
    uniquesmi = set()
    for line in data[1:]:
        year, aid, smi, ik, proteinid, bao, endpoint, val = line.split("\n")[0].split(",")
        uniquesmi.add((smi, ik))    
        if int(year) < 2016:
            if aid not in chronological_train:
                chronological_train[aid] = []
            chronological_train[aid].append((ik, proteinid, bao, endpoint, val))        
        else:
            if aid not in chronological_test:
                chronological_test[aid] = []
            chronological_test[aid].append((ik, proteinid, bao, endpoint, val))

        if aid not in assay_group:
            assay_group[aid] = []
        assay_group[aid].append((ik, proteinid, bao, endpoint, val))
 
    json.dump(assay_group, open("../data/assay_data.json","w"))
    # We do not aggregate test data, keep the assay and original value 
    # Aim to show the heterogeneity issue
    chronological_test_dict = {}
    for key, item in chronological_test.items():
        chronological_test_dict[key] = [(i[0], i[1], float(i[-1])) for i in item]
    json.dump(chronological_test_dict,  open("../data/Chronological/chronological_test.json","w"))

    
    # Aggregated data
    # Note we only aggregate data in training set. 
    # We keep the assay with their original values in test set 
    pl_group_chronological = {}
    pl_group_chronological_cell, pl_group_chronological_biochemical = {}, {}
    for key, item in chronological_train.items():
        for i in item:
            ik, proteinid, bao, endpoint, val = i
            plkey = (ik, proteinid)
            if plkey not in pl_group_chronological:
                pl_group_chronological[plkey] = []
            pl_group_chronological[plkey].append(float(val))
            
            if bao == "BAO_0000219":
                if plkey not in pl_group_chronological_cell:
                    pl_group_chronological_cell[plkey] = []
                pl_group_chronological_cell[plkey].append(float(val))
            if bao in ["BAO_0000357", "BAO_0000224"]:
                if plkey not in pl_group_chronological_biochemical:
                    pl_group_chronological_biochemical[plkey] = []
                pl_group_chronological_biochemical[plkey].append(float(val))


    # Chronological Full
    aggregated_train_chronological = open("../data/Aggregated/chronological_train.txt","w")
    for plkey, plitem in pl_group_chronological.items():
        if len(plitem) >= 2: # contain duplicates
            logplitem = np.log10(plitem)
            if max(logplitem) - min(logplitem) <= 0.3: # i.e. less than 3 fold
                aggregated_values = gmean(plitem)
                aggregated_train_chronological.write("{},{},{}\n".format(*plkey, aggregated_values))
        else:
            aggregated_values = plitem[0]
            aggregated_train_chronological.write("{},{},{}\n".format(*plkey, aggregated_values))
    aggregated_train_chronological.close()

    # Chronological Cell-Based Only
    aggregated_train_chronological_cell = open("../data/Aggregated/chronological_train_cell.txt","w")
    for plkey, plitem in pl_group_chronological_cell.items():
        if len(plitem) >= 2: # contain duplicates
            logplitem = np.log10(plitem)
            if max(logplitem) - min(logplitem) <= 0.3: # i.e. less than 3 fold
                aggregated_values = gmean(plitem)
                aggregated_train_chronological_cell.write("{},{},{}\n".format(*plkey, aggregated_values))
        else:
            aggregated_values = plitem[0]
            aggregated_train_chronological_cell.write("{},{},{}\n".format(*plkey, aggregated_values))
    aggregated_train_chronological_cell.close()

    # Chronological Cell-Based Only
    aggregated_train_chronological_biochemical = open("../data/Aggregated/chronological_train_biochemical.txt","w")
    for plkey, plitem in pl_group_chronological_biochemical.items():
        if len(plitem) >= 2: # contain duplicates
            logplitem = np.log10(plitem)
            if max(logplitem) - min(logplitem) <= 0.3: # i.e. less than 3 fold
                aggregated_values = gmean(plitem)
                aggregated_train_chronological_biochemical.write("{},{},{}\n".format(*plkey, aggregated_values))
        else:
            aggregated_values = plitem[0]
            aggregated_train_chronological_biochemical.write("{},{},{}\n".format(*plkey, aggregated_values))
    aggregated_train_chronological_biochemical.close()

    # Meta Chronological
    meta_train_chronological, meta_train_chronological_biochemical, meta_train_chronological_cell = {}, {}, {}
    for key, item in chronological_train.items():
        ik, proteinid, bao, endpoint, val = item[0]
        meta_train_chronological[key] = [(i[0], i[1], i[-1]) for i in item]
        if bao == "BAO_0000219":
            meta_train_chronological_cell[key] = [(i[0], i[1], i[-1]) for i in item]
        if bao in ["BAO_0000357", "BAO_0000224"]:
            meta_train_chronological_biochemical[key] = [(i[0], i[1], i[-1]) for i in item]

    json.dump(meta_train_chronological, open("../data/Chronological/chronological_train.json","w"))
    json.dump(meta_train_chronological_biochemical, open("../data/Chronological/chronological_train_biochemical.json","w"))
    json.dump(meta_train_chronological_cell, open("../data/Chronological/chronological_train_cell.json","w"))

    # Paired Assay
    pl_unpaired, pl_group_unpaired = {}, {}
    pl_unpaired_cell, pl_unpaired_biochemical = {}, {}
    
    for key, plitem in assay_group.items():
        if key not in paired_assay_aid:
            pl_group_unpaired[key] = plitem
            for i in plitem:
                ik, proteinid, bao, endpoint, val = i
                plkey = (ik, proteinid)
                if plkey not in pl_unpaired:
                    pl_unpaired[plkey] = []
                pl_unpaired[plkey].append(float(val))
        
            for i in plitem:
                ik, proteinid, bao, endpoint, val = i
                if bao == "BAO_0000219": # Cell Based
                    plkey = (ik, proteinid)
                    if plkey not in pl_unpaired_cell:
                        pl_unpaired_cell[plkey] = []
                    pl_unpaired_cell[plkey].append(float(val))
                if bao in ["BAO_0000357", "BAO_0000224"]:
                    plkey = (ik, proteinid)
                    if plkey not in pl_unpaired_biochemical:
                        pl_unpaired_biochemical[plkey] = []
                    pl_unpaired_biochemical[plkey].append(float(val))
    
    aggregated_train_paired = open("../data/Aggregated/unpaired_train.txt","w")
    for plkey, plitem in pl_unpaired.items():
        if len(plitem) >= 2: # contain duplicates
            logplitem = np.log10(plitem)
            if max(logplitem) - min(logplitem) <= 0.3: # i.e. less than 3 fold
                aggregated_values = gmean(plitem)
                aggregated_train_paired.write("{},{},{}\n".format(*plkey, aggregated_values))
        else:
            aggregated_values = plitem[0]
            aggregated_train_paired.write("{},{},{}\n".format(*plkey, aggregated_values))
    aggregated_train_paired.close()

    aggregated_train_paired_cell = open("../data/Aggregated/unpaired_train_cell.txt","w")
    for plkey, plitem in pl_unpaired_cell.items():
        if len(plitem) >= 2: # contain duplicates
            logplitem = np.log10(plitem)
            if max(logplitem) - min(logplitem) <= 0.3: # i.e. less than 3 fold
                aggregated_values = gmean(plitem)
                aggregated_train_paired_cell.write("{},{},{}\n".format(*plkey, aggregated_values))
        else:
            aggregated_values = plitem[0]
            aggregated_train_paired_cell.write("{},{},{}\n".format(*plkey, aggregated_values))
    aggregated_train_paired_cell.close()

    
    aggregated_train_paired_biochemical = open("../data/Aggregated/unpaired_train_biochemical.txt","w")
    for plkey, plitem in pl_unpaired_biochemical.items():
        if len(plitem) >= 2: # contain duplicates
            logplitem = np.log10(plitem)
            if max(logplitem) - min(logplitem) <= 0.3: # i.e. less than 3 fold
                aggregated_values = gmean(plitem)
                aggregated_train_paired_biochemical.write("{},{},{}\n".format(*plkey, aggregated_values))
        else:
            aggregated_values = plitem[0]
            aggregated_train_paired_biochemical.write("{},{},{}\n".format(*plkey, aggregated_values))
    aggregated_train_paired_biochemical.close()

    # Meta Paired
    meta_train_paired, meta_train_paired_biochemical, meta_train_paired_cell = {}, {}, {}
    for key, item in pl_group_unpaired.items():
        ik, proteinid, bao, endpoint, val = item[0]
        meta_train_paired[key] = [(i[0], i[1], i[-1]) for i in item]
        if bao == "BAO_0000219":
            meta_train_paired_cell[key] = [(i[0], i[1], i[-1]) for i in item]
        if bao in ["BAO_0000357", "BAO_0000224"]:
            meta_train_paired_biochemical[key] = [(i[0], i[1], i[-1]) for i in item]

    json.dump(meta_train_paired, open("../data/PairedSplit/unpaired_train.json","w"))
    json.dump(meta_train_paired_biochemical, open("../data/PairedSplit/unpaired_train_biochemical.json","w"))
    json.dump(meta_train_paired_cell, open("../data/PairedSplit/unpaired_train_cell.json","w"))
        

    # Generate Paired Assay Test Set
    paired_assay_test = {}
    for iidx, aid_i in enumerate(paired_assay_aid):
        ai = assay_group[aid_i]
        for jidx, aid_j in enumerate(paired_assay_aid):
            if aid_j != aid_i and jidx > iidx:
                aj = assay_group[aid_j]
                if aj[0][1] == ai[0][1]: # i.e. identical target
                    aj_ik, ai_ik = [x[0] for x in ai], [x[0] for x in aj]
                    common = set(aj_ik).intersection(set(ai_ik))
                    if len(common) >= 15:
                        ai_subset, aj_subset = sorted([(x[0], x[1], float(x[-1])) for x in ai if x[0] in common]), sorted([(x[0], x[1], float(x[-1])) for x in aj if x[0] in common])
                        ai_lvalues, aj_lvalues = np.log10([x[2] for x in ai_subset]), np.log10([x[2] for x in aj_subset])
                        if np.abs(ai_lvalues - aj_lvalues).mean()>0.5:
                            paired_assay_test["--".join([aid_i, aid_j])] = [ai_subset, aj_subset]
                    
    json.dump(paired_assay_test, open("../data/PairedSplit/paired_test.json","w"))

    """
    # Generate molecular graph
    if not os.path.exists("../data/molecule"):
        os.mkdir("../data/molecule")

    molecule_featurizer = molfeaturizer.MolFeaturizer(stereochemistry=True)
    for (smi, ik) in uniquesmi:
        mol = Chem.MolFromSmiles(smi)
        node, edge, index = molecule_featurizer.featurize(mol)
        src, tgt = list(zip(*index))
        graph = dgl.graph((torch.tensor(src), torch.tensor(tgt)))
        graph.ndata["h"] = torch.from_numpy(np.array(node)).type(torch.FloatTensor)
        pickle.dump(graph, open("../data/molecule/{}.pkl".format(ik), "wb"))

    # Generate sequence features
    if not os.path.exists("../data/sequence"):
        os.mkdir("../data/sequence")

    sequence = json.load(open("../data/sequence_dictionary.json","r"))
    sequence_feature = {}
    
    sequence_featurizer = seqfeaturizer.SequenceSWFeaturizer(max_sequence_length = 1500, sequence_type="AA-Special")
    
    for key, item in sequence.items():
        seqidx = item
        binary_feature, dct_feature = sequence_featurizer.featurize(key)
        sequence_feature[seqidx] = dct_feature
    pickle.dump(sequence_feature, open("../data/sequence/sequence_feature_dictionary.pkl","wb"))
    """
        
if __name__ == "__main__":
    main()
