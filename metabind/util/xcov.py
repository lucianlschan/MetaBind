from collections import Counter
import numpy as np
import torch
from torch import nn

def cross_covariance_aggregate(x, y, index):
    output, size = [], []
    key_position = {}
    for pos, i in enumerate(index):
        if i not in key_position:
            key_position[i] = []
        key_position[i].append(pos)

    for key in sorted(key_position.keys()):
        positions = key_position[key]
        x_arr = x[[positions]]
        y_arr = y[[positions]]
        exy = (torch.matmul(x_arr.transpose(0,1), y_arr)/len(positions)).flatten().unsqueeze(0)  # flatten cross covariance matrix
        output.append(exy)
        size.append(len(key_position[key]))
    return torch.concat(output, dim=0), size

