from collections import Counter
import numpy as np
from scipy.fftpack import dct
import torch
from torch import nn


AACharacter = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
ambigiuous = ["B","Z"] # Ambigiuous (B: D/N, Z: E/Q)
NucleotideCharacter = ["A","C","G","T","U"]
NucleotideComplementary = ["T","G","C","A","A"]
gap = ["-"]
special = ["U","X"] # X Unknown, U: Selenocystenine

sidechain_group = {"NoSideChain": ["G"],
            "Branched": ["B","D","E","I","L","N","Q","R","T","V","Z"],
            "Unbranched": ["A","C","K","M","S"],
            "Cyclic": ["F", "H", "P", "W", "Y"]}

residue_group = {"Small": ["A","G"],
                "Hydrophobic":["I","L","M","P","V"],
                "Aromatic": ["F","W","Y"],
                "Polar":["C","N","Q","S","T"],
                "Postive Charged":["H","K","R"],
                "Negative Charged":["D","E"]}

def slidingwindow(seq, window_size=20, length=2000):
    seqsize = len(seq)
    if seqsize > length:
        raise ValueError("Sequence length exceed maximum length")
    if seqsize - window_size > 0:
        for i in range(0, seqsize-window_size):
            yield seq[i:i+window_size]
    else:
        yield seq

class SequenceSWFeaturizer:
    def __init__(self,
                max_sequence_length = 1000,
                window_size = 1,
                padding_type = "Extreme",
                sequence_type = "AA-Standard",
                sidechain = True,
                residue = True,
                discrete_cosine_transform = True):
        assert padding_type in ["Extreme","Pre", "Post"]
        self.maxlen = max_sequence_length
        self.windowsize = window_size
        self.padding_type = padding_type
        self.seqtype = sequence_type
        self.sidechain = sidechain
        self.residue = residue_group
        self.sidechain_group_key = list(sidechain_group.keys())
        self.residue_group_key = list(residue_group.keys())
        self.dct = discrete_cosine_transform
        if self.seqtype == "AA-Only":
            self.aachar = AACharacter
        elif self.seqtype == "AA-Standard":
            self.aachar = AACharacter
        elif self.seqtype == "AA-Special":
            self.aachar = AACharacter + special
        elif self.seqtype == "AA-Full":
            self.aachar =  AACharacter + ambigiuous + gap + special
        else:
            raise NotImplementedError
        self.nucleotidechar = NucleotideCharacter + gap


    def featurize(self, seq):
        seqlen = len(seq)
        assert seqlen <= self.maxlen
        lendiff = self.maxlen - seqlen
        pre_padding = lendiff
        extreme_padding = int(lendiff/2) if lendiff%2 == 0 else int((lendiff-1)/2)
        nrow = self.maxlen - self.windowsize + 1
        seqtensor, sidechaintensor, residuetensor = None, None, None
        if self.seqtype.startswith("AA"):
            seqtensor = torch.zeros([nrow, len(self.aachar)])
            if self.sidechain:
                sidechaintensor = torch.zeros([nrow, len(self.sidechain_group_key)])
            if self.residue:
                residuetensor = torch.zeros([nrow, len(self.residue_group_key)])
        elif self.seqtype == "Nucleotide":
            seqtensor = torch.zeros([nrow, len(self.nucleotidechar)])
        else:
            raise NotImplementedError

        swindows = slidingwindow(seq,
                                window_size=self.windowsize,
                                length=self.maxlen)

        for idx, sw in enumerate(swindows):
            for key, count in Counter(sw).items():
                if self.seqtype.startswith("AA"):
                    if self.padding_type == "Post":
                        seqtensor[idx, self.aachar.index(key)] = count/self.windowsize # Sequence----
                    elif self.padding_type == "Pre":
                        seqtensor[idx + pre_padding, self.aachar.index(key)] = count/self.windowsize # --> -------Sequence
                    else:
                        seqtensor[idx + extreme_padding, self.aachar.index(key)] = count/self.windowsize  # -->  ----Sequence----

                    # Sidechain Features
                    if self.sidechain:
                        for skey, sgroup in sidechain_group.items():
                            if key in sgroup:
                                if self.padding_type == "Post":
                                    sidechaintensor[idx, self.sidechain_group_key.index(skey)] = count/self.windowsize
                                elif self.padding_type == "Pre":
                                    sidechaintensor[idx + pre_padding, self.sidechain_group_key.index(skey)] = count/self.windowsize
                                else:
                                    sidechaintensor[idx + extreme_padding, self.sidechain_group_key.index(skey)] = count/self.windowsize

                    if self.residue:
                        for rkey, rgroup in residue_group.items():
                            if key in rgroup:
                                if self.padding_type == "Post":
                                    residuetensor[idx, self.residue_group_key.index(rkey)] = count/self.windowsize
                                elif self.padding_type == "Pre":
                                    residuetensor[idx + pre_padding, self.residue_group_key.index(rkey)] = count/self.windowsize
                                else:
                                    residuetensor[idx + extreme_padding, self.residue_group_key.index(rkey)] = count/self.windowsize

                else:
                    seqtensor[idx, self.nucleotidechar.index(key)] = count/self.windowsize

        if sidechaintensor is not None:
            seqtensor = torch.concat([seqtensor, sidechaintensor], dim=1)
        if residuetensor is not None:
            seqtensor = torch.concat([seqtensor, residuetensor], dim=1)
        if self.dct:
            seqtensor_npy = seqtensor.numpy()
            dct_seqtensor = dct(dct(seqtensor_npy.T, norm="ortho").T, norm="ortho")
            return seqtensor, torch.tensor(dct_seqtensor)
        else:
            return seqtensor

