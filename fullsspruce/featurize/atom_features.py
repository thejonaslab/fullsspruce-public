"""
Per-atom features : Featurizations that return one per atom
[in contrast to whole-molecule featurizations]
"""

import numpy as np
import torch
from rdkit import Chem
from fullsspruce import util
from fullsspruce.util import get_nos

HYBRIDIZATIONS = [Chem.HybridizationType.UNSPECIFIED,
                  Chem.HybridizationType.S, 
                  Chem.HybridizationType.SP, 
                  Chem.HybridizationType.SP2, 
                  Chem.HybridizationType.SP3, 
                  Chem.HybridizationType.SP3D, 
                  Chem.HybridizationType.SP3D2,
                  Chem.HybridizationType.OTHER
                  ]

CHI_TYPES = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED, 
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]

# HYBRIDIZATIONS = list(Chem.HybridizationType.values.values())
# CHI_TYPES = list(Chem.rdchem.ChiralType.values.values())


def to_onehot(x, vals):
    return [x == v for v in vals]
        
MMFF94_ATOM_TYPES = [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 
                      11, 12, 15, 16, 17, 18, 20,
                      21, 22, 23, 24, 25, 26, 27, 28, 29, 
                      30, 31, 32, 33, 37, 38, 39, 40,
                      42, 43, 44, 46, 48, 59, 62, 63, 64, 
                      65, 66, 70, 71, 72, 74, 75, 78]


ELECTRONEGATIVITIES = {1: 2.20,
                       6: 2.26,
                       7: 3.04,
                       8: 3.44,
                       9: 3.98,
                       15: 2.19,
                       16: 2.58,
                       17: 3.16}
                       
                       
def feat_tensor_atom(mol, 
                     feat_atomicno = True, feat_pos=True, 
                     feat_atomicno_onehot=[1, 6, 7, 8, 9], 
                     feat_valence=True, aromatic=True, 
                     hybridization=True, 
                     partial_charge=True, formal_charge=True, 
                     r_covalent=True,
                     r_vanderwals=True, default_valence=True, 
                     rings=False, 
                     total_valence_onehot=False, 
                     mmff_atom_types_onehot=False,
                     max_ring_size=8,
                     rad_electrons = False,
                     chirality = False,
                     assign_stereo=False,
                     electronegativity= False,
                     DEBUG_fchl=False):

    """
    Featurize a molecule on a per-atom basis
    feat_atomicno_onehot : list of atomic numbers

    Always assume using conf_idx unless otherwise passed

    Returns an (ATOM_N x feature) float32 tensor

    NOTE: Performs NO santization or cleanup of molecule, 
    assumes all molecules have sanitization calculated ahead
    of time. 

    """

    pt = Chem.GetPeriodicTable()
    mol = Chem.Mol(mol) # copy molecule

    atomic_nos = get_nos(mol)   

    if partial_charge:
        Chem.rdPartialCharges.ComputeGasteigerCharges(mol)

    atom_features = []
    if mmff_atom_types_onehot:
        mmff_p = Chem.rdForceFieldHelpers.MMFFGetMoleculeProperties(mol)

    if assign_stereo:
        Chem.rdmolops.AssignStereochemistryFrom3D(mol)

    # if DEBUG_fchl:
    #     import qml
    
    #     fchl_rep = qml.fchl.generate_representation(coords, atomic_nos, max_size=64)
    #     fchl_rep = fchl_rep.reshape(fchl_rep.shape[0], -1)
        
    for i in range(mol.GetNumAtoms()):
        a = mol.GetAtomWithIdx(i)
        atomic_num = int(atomic_nos[i])
        atom_feature = []

        if feat_atomicno:
            atom_feature += [atomic_num]
        # if feat_pos:
        #     atom_feature += coords[i].tolist()

        if feat_atomicno_onehot is not None :
            atom_feature += to_onehot(atomic_num, feat_atomicno_onehot)
        if feat_valence:
            atom_feature += [a.GetTotalValence()]
        if total_valence_onehot:
            atom_feature +=  to_onehot(a.GetTotalValence(), range(1, 7))
        if aromatic:
            atom_feature += [a.GetIsAromatic()]
        if hybridization:
            atom_feature += to_onehot(a.GetHybridization(), HYBRIDIZATIONS)
        if partial_charge:
            gc = float(a.GetProp('_GasteigerCharge'))
            #assert np.isfinite(gc)
            if not np.isfinite(gc):
                gc = 0.0
            atom_feature += [gc]
        if formal_charge:
            atom_feature += to_onehot(a.GetFormalCharge(), [-1, 0, 1])
        if r_covalent:
            atom_feature += [pt.GetRcovalent(atomic_num)]
        if r_vanderwals:
            atom_feature += [pt.GetRvdw(atomic_num)]

        if default_valence:
            atom_feature += to_onehot(pt.GetDefaultValence(atomic_num), range(1, 7))
        if rings:
           atom_feature +=  [a.IsInRingSize(r) for r in range(3, max_ring_size)]
        if rad_electrons:
            if a.GetNumRadicalElectrons() > 0:
                raise ValueError("RADICAL")
        if chirality:
            atom_feature += to_onehot(a.GetChiralTag(),
                                      CHI_TYPES)
        if electronegativity:
            atom_feature += [ELECTRONEGATIVITIES[atomic_num]]
        if mmff_atom_types_onehot:
            if mmff_p is None:
                atom_feature += [0] * len(MMFF94_ATOM_TYPES)
            else:
                atom_feature += to_onehot(mmff_p.GetMMFFAtomType(i), 
                                          MMFF94_ATOM_TYPES)
        if DEBUG_fchl:
            fchl_val = fchl_rep[i]
            fchl_val[fchl_val > 1e5] = 0
            #print(fchl_val.shape, fchl_val)
            #assert np.isfinite(fchl_val).all()
            
            atom_feature += fchl_val.tolist()
        atom_features.append(atom_feature)
    return torch.Tensor(atom_features)

