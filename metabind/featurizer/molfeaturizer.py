import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

"""
Node Features
Atom Type: C, N, O, F, P, S, Cl, Br, I
Formal Charge: Int
Hybridization: s, sp, sp2, sp3, sp3d, sp3d2, unspecified
Aromatic: True/False
Degree: Int (0-5)
Number of Hydrogens:

Edge Features:
Bond Type: single, double, triple, aromatic
Same Ring: pair of atoms in the same ring
Conjugated: True/False
Stereo: Stereo Configuration of bond

"""

Ptable = Chem.GetPeriodicTable()

class MolFeaturizer:
    def __init__(self,
                max_num_atoms=50,
                kekulize=True,
                stereochemistry = True,
                atom_basic_feature = True,
                edge_basic_feature = True):

        self.maxnatoms = max_num_atoms
        self.kekulize = kekulize
        self.maxdegree = 6
        self.stereochemistry = stereochemistry
        self.atom_basic_feature = atom_basic_feature
        self.edge_basic_feature = edge_basic_feature
        # Basic Atom Features
        self.symbols = ["C","N","O","F","P","S","Cl","Br","I"]
        self.hybridizations = [Chem.rdchem.HybridizationType.S,
                Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3,
                Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2,
                Chem.rdchem.HybridizationType.UNSPECIFIED]

        # Basic Bond Features
        self.bondtype = [Chem.rdchem.BondType.SINGLE,
                        Chem.rdchem.BondType.DOUBLE,
                        Chem.rdchem.BondType.TRIPLE,
                        Chem.rdchem.BondType.AROMATIC]



    def featurize(self, mol):
        assert mol.GetNumAtoms() <= self.maxnatoms, "Molecule contains more than {} atoms.".format(self.maxnatoms)

        # Require pre-specified stereochemistry in molecule.
        if self.stereochemistry:
            # R/S Stereo
            chirality = dict(Chem.FindMolChiralCenters(mol))
            # E/Z Stereo
            atomEZ = {}
            bondstereo = {}
            for bond in mol.GetBonds():
                if str(bond.GetStereo()) == "STEREOE":
                    bondstereo[(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())] = "E"
                    bondstereo[(bond.GetEndAtomIdx(), bond.GetBeginAtomIdx())] = "E"
                    atomEZ[bond.GetBeginAtomIdx()] = "E"
                    atomEZ[bond.GetEndAtomIdx()] = "E"
                elif str(bond.GetStereo()) == "SETREOZ":
                    bondstereo[(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())] = "Z"
                    bondstereo[(bond.GetEndAtomIdx(), bond.GetBeginAtomIdx())] = "Z"
                    atomEZ[bond.GetBeginAtomIdx()] = "Z"
                    atomEZ[bond.GetEndAtomIdx()] = "Z"

        node_features, edge_features, edge_indices = [], [], []

        if self.atom_basic_feature:
            # Node (Atom Features)
            node_basic = []
            for atom in mol.GetAtoms():
                symbol = [0.]*len(self.symbols)
                degree = [0.]*(self.maxdegree+1)
                hybridizations = [0]*len(self.hybridizations)
                hydrogens = [0.]*5
                if atom.GetSymbol() in self.symbols: symbol[self.symbols.index(atom.GetSymbol())] = 1.
                if atom.GetDegree() <= self.maxdegree: degree[atom.GetDegree()] = 1.
                hybridizations[self.hybridizations.index(atom.GetHybridization())] = 1.
                aromaticity = 1. if atom.GetIsAromatic() else 0.
                hydrogens[atom.GetTotalNumHs()] = 1.
                formalcharge = atom.GetFormalCharge()
                stereos = [0.]*4 # R,S,E,Z
                if self.stereochemistry:
                    rs = chirality.get(atom.GetIdx(), "None")
                    if rs == "R":
                        stereos[0] = 1.
                    elif rs == "S":
                        stereos[1] = 1.
                    ez = atomEZ.get(atom.GetIdx(), "None")
                    if ez == "E":
                        stereos[2] = 1.
                    elif ez == "Z":
                        stereos[3] = 1.

                node = symbol + degree + hybridizations + [aromaticity] + [formalcharge] + hydrogens + \
                       stereos
                node_features.append(node)

        edge_basic = []
        for bond in mol.GetBonds():
            edge_indices.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
            edge_indices.append((bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()))
            if self.edge_basic_feature:
                bond_type = [0.]*len(self.bondtype)
                bond_type[self.bondtype.index(bond.GetBondType())] = 1.
                conjugation = 1. if bond.GetIsConjugated() else 0.
                ring = 1. if bond.IsInRing() else 0.
                edge = bond_type + [conjugation, ring]

                edge_basic.append(edge)
                edge_basic.append(edge)

        if self.edge_basic_feature:
            edge_basic = np.array(edge_basic, dtype=np.float32)
            edge_features.append(edge_basic)
    
        return node_features, edge_features, edge_indices
