import os
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
import pickle
import rdkit
import scipy
import pandas as pd
import itertools
import time
import torch
import io
import zlib

import collections
import scipy.optimize
import scipy.special
import scipy.spatial.distance
from fullsspruce.model import nets
from tqdm import tqdm

import tinygraph.io.rdkit
import tinygraph as tg

import warnings

Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

nuc_to_atomicno = {'13C' : 6, 
                   '1H' : 1}

EV_TO_KCAL_MOL = 23.06035


def array_to_conf(mat):
    """
    Take in a (N, 3) matrix of 3d positions and create
    a conformer for those positions. 
    
    ASSUMES atom_i = row i so make sure the 
    atoms in the molecule are the right order!
    
    """
    N = mat.shape[0]
    conf = Chem.Conformer(N)
    
    for ri in range(N):
        p = rdkit.Geometry.rdGeometry.Point3D(*mat[ri])                                      
        conf.SetAtomPosition(ri, p)
    return conf


def numpy(x):
   """
   pytorch convenience method just to get a 
   numpy array back from a tensor or variable
   """
   if isinstance(x, np.ndarray):
      return x
   if isinstance(x, list):
      return np.array(x)

   if isinstance(x, torch.Tensor):
      if x.is_cuda:
         return x.cpu().numpy()
      else:
         return x.numpy()
   raise NotImplementedError(str(type(x)))

def get_nos_coords(mol, conf_i):
    conformer = mol.GetConformers()[conf_i]
    coord_objs = [conformer.GetAtomPosition(i) for i in  range(mol.GetNumAtoms())]
    coords = np.array([(c.x, c.y, c.z) for c in coord_objs])
    atomic_nos = np.array([a.GetAtomicNum() for a in mol.GetAtoms()]).astype(int)
    return atomic_nos, coords

def get_nos(mol):
    return np.array([a.GetAtomicNum() for a in mol.GetAtoms()]).astype(int)

def recursive_update(d, u):
    ### Dict recursive update 
    ### https://stackoverflow.com/a/3233356/1073963
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def morgan4_crc32(m):
   mf = Chem.rdMolDescriptors.GetHashedMorganFingerprint(m, 4)
   crc = zlib.crc32(mf.ToBinary())
   return crc


def get_atom_counts(rdmol):
    counts = {}
    for a in rdmol.GetAtoms():
        s = a.GetSymbol()
        if s not in counts:
            counts[s] = 0
        counts[s] += 1
    return counts

def get_ring_size_counts(rdmol):
    counts = {}
    ssr = Chem.rdmolops.GetSymmSSSR(rdmol)
    for ring_members in ssr:
        rs = len(ring_members)
        rs_str = rs
        
        if rs_str not in counts:
            counts[rs_str]=0
        counts[rs_str] += 1
    return counts    


def filter_mols(mol_dicts, filter_params, 
                other_attributes = [],
                sanitize=False, 
                mol_from_binary=False):
    """
    Filter molecules per criteria
    """
   
    skip_reason = []
    ## now run the query
    output_mols = []
    for row in tqdm(mol_dicts):
        mol_id = row['id']
        if not mol_from_binary:
            mol = Chem.Mol(row['mol'])
        else:
            mol = Chem.Mol(zlib.decompress(row['bmol']))
        if mol is None:
            print(row)
        if sanitize:
           Chem.SanitizeMol(mol)
        atom_counts = get_atom_counts(mol)
        if not set(atom_counts.keys()).issubset(filter_params['elements']):
            skip_reason.append({'mol_id' : mol_id, 'reason' : "elements"})
            continue

        if mol.GetNumAtoms() > filter_params['max_atom_n']:
            skip_reason.append({'mol_id' : mol_id, 'reason' : "max_atom_n"})
            continue

        if mol.GetNumHeavyAtoms() > filter_params['max_heavy_atom_n']:
            skip_reason.append({'mol_id' : mol_id, 'reason' : "max_heavy_atom_n"})
            continue

        ring_size_counts = get_ring_size_counts(mol)
        if len(ring_size_counts) > 0:
            if np.max(list(ring_size_counts.keys())) > filter_params['max_ring_size']:
                skip_reason.append({'mol_id' : mol_id, 'reason' : "max_ring_size"})
                continue
            if np.min(list(ring_size_counts.keys())) < filter_params['min_ring_size']:
                skip_reason.append({'mol_id' : mol_id, 'reason' : "min_ring_size"})
                continue
        skip_mol = False
        for a in mol.GetAtoms():
            if a.GetFormalCharge() != 0 and not filter_params['allow_atom_formal_charge'] :
                skip_mol = True
                skip_reason.append({'mol_id' : mol_id, 'reason' : "atom_formal_charge"})

                break

            if a.GetHybridization() == 0 and not filter_params['allow_unknown_hybridization'] :
                skip_mol = True
                skip_reason.append({'mol_id' : mol_id, 'reason' : "unknown_hybridization"})

                break
            if a.GetNumRadicalElectrons() > 0 and not filter_params['allow_radicals'] :
                skip_mol = True
                skip_reason.append({'mol_id' : mol_id, 'reason' : "radical_electrons"})

                break
        if skip_mol:
            continue

        if Chem.rdmolops.GetFormalCharge(mol) != 0 and not filter_params['allow_mol_formal_charge']:
            skip_reason.append({'mol_id' : mol_id, 'reason' : "mol_formal_charge"})

            continue

        skip_reason.append({'mol_id' : mol_id, 'reason' : None})
        
        out_row = {'molecule_id' : mol_id, 
                #    'mol': mol.ToBinary(), 
                   # 'source' : row['source'],  # to ease downstream debugging
                   # 'source_id' : row['source_id'], 
                   'simple_smiles' : Chem.MolToSmiles(Chem.RemoveHs(mol), 
                                                      isomericSmiles=False)
                   }
        for f in other_attributes:
           out_row[f] = row[f]

        output_mols.append(out_row)
    output_mol_df = pd.DataFrame(output_mols)
    skip_reason_df = pd.DataFrame(skip_reason)
    return output_mol_df, skip_reason_df


PERM_MISSING_VALUE = 1000    
def vect_pred_min_assign(pred, y, mask, Y_MISSING_VAL=PERM_MISSING_VALUE):    
    true_vals = y # [mask>0] 
    true_vals = true_vals[true_vals < Y_MISSING_VAL]

    dist = scipy.spatial.distance.cdist(pred.reshape(-1, 1), true_vals.reshape(-1, 1))
    dist[mask == 0] = 1e5
    ls_assign = scipy.optimize.linear_sum_assignment(dist)
    mask_out = np.zeros_like(mask)
    y_out = np.zeros_like(y)

    for i, o in zip(*ls_assign):
        mask_out[i] = 1
        y_out[i] = true_vals[o]
    
    return y_out, mask_out


def min_assign(pred, y, mask, Y_MISSING_VAL=PERM_MISSING_VALUE):
    """
    Find the minimum assignment of y to pred
    
    pred, y, and mask are (BATCH, N, 1) but Y is unordered and
    has missing entries set to Y_MISSING_VAL 

    returns a new y and pred which can be used
    """
    BATCH_N, _ = pred.shape
    if pred.ndim > 2:
       pred = pred.squeeze(-1)
       y = y.squeeze(-1)
       mask = mask.squeeze(-1)
    
    y_np = y.cpu().detach().numpy()
    mask_np = mask.numpy()
    # print("total mask=", np.sum(mask_np))
    pred_np = pred.numpy()
    
    out_y_np = np.zeros_like(y_np)
    out_mask_np = np.zeros_like(pred_np)
    for i in range(BATCH_N):
       # print("batch_i=", i, pred_np[i], 
       #       y_np[i], 
       #       mask_np[i])
       out_y_np[i], out_mask_np[i] = vect_pred_min_assign(pred_np[i], 
                                                           y_np[i], 
                                                           mask_np[i], Y_MISSING_VAL)
    
    out_y = torch.Tensor(out_y_np)
    out_mask = torch.Tensor(out_mask_np)
    if torch.sum(mask) > 0:
       assert torch.sum(out_mask) > 0
    return out_y, out_mask 

  
def mol_with_atom_index( mol, make2d=True):
    "For plotting"
    mol = Chem.Mol(mol)
    if make2d:
       Chem.AllChem.Compute2DCoords(mol)
    atoms = mol.GetNumAtoms()
    for idx in range( atoms ):
        mol.GetAtomWithIdx( idx ).SetProp( 'molAtomMapNumber', str( mol.GetAtomWithIdx( idx ).GetIdx() ) )
    return mol


def kcal_to_p(energies, T=298):
    k_kcal_mol = 0.001985875 # kcal/(mol⋅K)

    es_kcal_mol = np.array(energies)
    log_pstar = -es_kcal_mol/(k_kcal_mol * T)
    pstar = log_pstar - scipy.special.logsumexp(log_pstar)
    p = np.exp(pstar)
    p = p / np.sum(p)
    return p 

def get_methyl_hydrogens(m):
    """
    returns list of (carbon index, list of methyl Hs)
    
    
    Originally in nmrabinitio
    """
    
    for c in m.GetSubstructMatches(Chem.MolFromSmarts("[CH3]")):
        yield c[0], [a.GetIdx() for a in m.GetAtomWithIdx(c[0]).GetNeighbors() if a.GetSymbol() == 'H']
        
def create_methyl_atom_eq_classes(mol):
    """
    Take in a mol and return an equivalence-class assignment vector
    of a list of frozensets

    Originally in nmrabinitio    
    """
    mh = get_methyl_hydrogens(mol)
    N = mol.GetNumAtoms()
    eq_classes = []
    for c, e in mh:
        eq_classes.append(frozenset(e))
    assert len(frozenset().intersection(*eq_classes)) == 0
    existing = frozenset().union(*eq_classes)
    for i in range(N):
        if i not in existing:
            eq_classes.append(frozenset([i]))
    return eq_classes
            
    
  


class EquivalenceClasses:
    """
    Equivalence classes of atoms and the kinds of questions
    we might want to ask. For example, treating all hydrogens
    in a methyl the same, or treating all equivalent atoms
    (from RDKit's perspective) the same. 
    
    Originally in nmrabinitio
    """
    def __init__(self, eq):
        """
        eq is a list of disjoint frozen sets of the partitioned
        equivalence classes. Note that every element must be
        in at least one set and there can be no gaps. 
        
        """
        
        
        all_elts = frozenset().union(*eq)
    
        N = np.max(list(frozenset().union(*eq))) + 1
        # assert all elements in set
        assert frozenset(list(range(N))) == all_elts

        self.eq = eq
        self.N = N
        
    def get_vect(self):
        assign_vect = np.zeros(self.N, dtype=int)
        for si, s in enumerate(sorted(self.eq, key=len)):
            for elt in s:
                assign_vect[elt] = si
        return assign_vect

    def get_pairwise(self):
        """
        From list of frozensets to all-possible pairwise assignment
        equivalence classes

        """
        eq = self.eq
        N = self.N
        
        assign_mat = np.ones((N, N), dtype=int)*-1
        eq_i = 0 
        for _, s1 in enumerate(sorted(eq, key=len)):
            for _, s2 in enumerate(sorted(eq, key=len)):
                for i in s1:
                    for j in s2:
                        assign_mat[i, j] = eq_i
                eq_i += 1
        assert (assign_mat != -1).all()
        return assign_mat

def methyl_average_coupling(mol, coupling_dict):
    """
    Computes the average couplings of equivalent methyls. 
    
    """
    eq_class_starts_at = mol.GetNumAtoms()
    eq_class_i = eq_class_starts_at
    eq_classes = {} # class to atom 
    eq_class_lut = {} # atom to class 
    for _, (c, hs) in enumerate(get_methyl_hydrogens(mol)):

        eq_classes[eq_class_i] = hs
        for h in hs:
            eq_class_lut[h] = eq_class_i
        eq_class_i += 1
    for i in range(mol.GetNumAtoms()):
        if i not in eq_class_lut:
            eq_class_lut[i] = i
            eq_classes[i] = [i]

    merged_coupling_dict = {}
    for (a1, a2), v in coupling_dict.items():

        new_key = (eq_class_lut[a1], eq_class_lut[a2])
        if new_key not in merged_coupling_dict:
            merged_coupling_dict[new_key] = []
        merged_coupling_dict[new_key].append(v)
    merged_coupling_dict = {k : np.mean(v) for k, v in merged_coupling_dict.items()}
    # unmerge dictionary 

    output_dict = {}
    for (eq1, eq2), v in merged_coupling_dict.items():
        for a1 in eq_classes[eq1]:
            for a2 in eq_classes[eq2]:
               if a1 != a2:
                  a, b = min(a1, a2), max(a1, a2)
                  output_dict[(a, b)] = v
    return output_dict     

def get_stacked_distance_mat(mol, conf_idxs=None):
    """
    Get all intra-atomic distances for mol conformers

    Returns the atomN x atomN x conformer numpy position 
    array for downstream processing. 

    Does not require confs to have unqiue IDs. 

    """

    dms = []
    for i, c in enumerate(mol.GetConformers()):
        if not conf_idxs is None and not i in conf_idxs:
            continue
        p = c.GetPositions() 
        dm = np.sqrt(np.sum((p[:, np.newaxis, :, ] - p[np.newaxis, :, :])**2, axis=-1))

        dms.append(dm)
    return np.stack(dms, -1) 


 
def get_conf_angles(mol, conf_idxs=None):
    """
    Get the distribution of angles between all A-B-C triples of atoms 
    across all conformers, in radians
    
    Returns atomN x atomN x confN matrix 
    0 otherwise. 
    
    Note that these will mostly be between 1.6 and 2.4 for the random
    linear C=C=C chain. 
    """
    confs = mol.GetConformers()
    confN = len(confs)
    if not conf_idxs is None:
        confN = len(conf_idxs)
    angles = np.zeros((mol.GetNumAtoms(), mol.GetNumAtoms(), confN))
    # zero is a fine sentinal value as you could never have a zero angle
    for a in mol.GetAtoms():
        neighbors = a.GetNeighbors()
        for a1, a2 in itertools.combinations(neighbors, 2):
            i, j = a1.GetIdx(), a2.GetIdx()
            i, j = min(i, j), max(i, j)
            ci = 0
            for iter_i, c in enumerate(confs):
                if not conf_idxs is None and not iter_i in conf_idxs:
                    continue
                angles[i, j, ci] = Chem.rdMolTransforms.GetAngleRad(c, i, a.GetIdx(), j)
                ci += 1

    
    return angles


def get_conf_dihedral_angles(mol, conf_idxs=None):
    """
    Get the distribution of angles between all A-B-C-D quads of atoms 
    across all conformers, in radians
    
    Returns atomN x atomN x confN matrix 
    0 otherwise. 
    """
    confs = mol.GetConformers()
    confN = len(confs)
    if not conf_idxs is None:
        confN = len(conf_idxs)
    angles = np.zeros((mol.GetNumAtoms(), mol.GetNumAtoms(), confN))
    # zero is a fine sentinal value as you could never have a zero angle

    tg_mol = tinygraph.io.rdkit.from_rdkit_mol(mol)
    sp, paths = tg.algorithms.get_shortest_paths(tg_mol, False, paths=True)

    for i, l in np.transpose((np.triu(sp) == 3).nonzero()): 
        # The path between i and l is chosen arbitrarily by tinygraph, but our assumption
        # is that this feature is mainly useful for coupling between protons, which can have
        # only one path. 
        i = int(i)
        l = int(l)
        j = int(paths[i][l])
        k = int(paths[j][l])
        ci = 0
        for iter_i, c in enumerate(confs):
            if not conf_idxs is None and not iter_i in conf_idxs:
                continue
            angles[i, l, ci] = Chem.rdMolTransforms.GetDihedralRad(c, i, j, k, l)
            ci += 1
    cs = []
    for c in range(confN):
        if np.isnan(angles[:,:,c]).any():
            cs += [c]
    angles = np.delete(angles, cs, 2)

    return angles


def enrich_confs_methyl_permute(mol, num_confs, random_seed=None):
    """
    For each conformer, add num new conformers that arise from permuting 
    the methyl hydrogens; this is a cheap monte-carlo way of dealing
    with methyl rotation. 
    
    num_confs: the number of additional conformers you want PER input conformer. 
    If the input mol has ten confs and num = 5 the output mol will have
    50 new conformers added. 

    Additionally we attempt to copy conformer properties, which
    in our case will often be energy values. 

    """

    mol = Chem.Mol(mol)
    methyl_eq_h = list(get_methyl_hydrogens(mol))
    if random_seed is not None:
        rng = np.random.default_rng(random_seed)
    else:
        rng = np.random.default_rng()

    positions = [c.GetPositions() for c in mol.GetConformers()]
    props = [c.GetPropsAsDict() for c in mol.GetConformers()]
    for ci, p in enumerate(positions):
        for _ in range(num_confs):
            new_positions = np.copy(p)

            for _, hs in methyl_eq_h:
                roll_amount = rng.integers(low=0, high=len(hs))
                new_idx = np.roll(hs, roll_amount)
                new_positions[hs] = p[new_idx]

            new_c = array_to_conf(new_positions)
            for k, v in props[ci].items():
                if isinstance(v, float):
                    new_c.SetDoubleProp(k, v)
                else:
                    raise NotImplementedError()
            mol.AddConformer(new_c, assignId=True)   
    return mol

def compute_conf_probs(conf_energies_ev, 
                        T = 273.15):
    """
    For conformer energies in EV compute the boltzmann-weighted
    probabilities at temp T=273.15 K 

    """
    
    EV_TO_KJ_MOL = 96.487
    R = 8.31446261815324 # J / K / mol 
    RkJ = R /1000
    
    energies_kjmol = conf_energies_ev * EV_TO_KJ_MOL
    min_e = np.min(energies_kjmol)
    energies_delta_kjmol = energies_kjmol - min_e
    pstar = np.exp(-energies_delta_kjmol/(RkJ*T))
    p = pstar / np.sum(pstar)
    return p

def assert_conf_ID_uniqueness(mol, conf_idxs=None):
    """
    Determine whether all conformers in a Mol have unique IDs. If they
    are all unique, return the list of IDs. If not, raise an Exception.
    If conf_idxs is not None, only consider the conformers with indices 
    in conf_idxs.
    """
    conf_IDs = []
    for i, c in enumerate(mol.GetConformers()):
        if not conf_idxs is None and not i in conf_idxs:
            continue
        conf_IDs += [c.GetId()]
    if len(conf_IDs) == len(set(conf_IDs)):
        return conf_IDs
    else:
        raise ValueError("Conformer IDs are not unique.")