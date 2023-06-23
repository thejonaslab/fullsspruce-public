import pandas as pd
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from fullsspruce.util import get_nos_coords
from fullsspruce import util
from fullsspruce.featurize.atom_features import to_onehot
import networkx as nx

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*') 

def canonical_rank_adj(m):
    """
    Create a matrix linking all atoms of equal canonical rank
    """
    r = np.array(Chem.rdmolfiles.CanonicalRankAtoms(m, breakTies=False))
    adj = np.array([np.array([1 if (rank2 == rank and not i == j) else 0 for j, rank2 in enumerate(r)]) for i, rank in enumerate(r)])
    return adj

def featurize_distance_matrix(distances, dist_feat_func, bins):
    """
    Create feature matrix based on input distances adjacency matrix
    """
    # Goal: Distances stored as n_atoms x n_atoms array
    # Use function distance_feature_func to create tensors
    a_d = []
    if dist_feat_func == 'inv':
        # dists = np.zeros_like(distances)
        # for i, row in enumerate(distances):
        #     for j, e in enumerate(row):
        #         if not e == 0:
        #             dists[i,j] = 1/e
        dists = 1/np.where(distances == 0, float('inf'), distances)
        a_d += [torch.tensor(dists)]
    elif dist_feat_func == 'bin':
        prev_b = 0
        for b in bins:
            # dists = np.zeros_like(distances)
            # for i, row in enumerate(distances):
            #     for j, e in enumerate(row):
            #         if e > prev_b and e <= b:
            #             # Should each edge only appear in one bin?
            #             dists[i,j] = 1
            dists = np.where((distances > prev_b) & (distances <= b), 1, 0)
            a_d += [torch.tensor(dists)]
            prev_b = b
    else:
        a_d = [torch.tensor(dists)]
    return a_d

def mol_to_nums_adj(m, MAX_ATOM_N=None):# , kekulize=False):
    """
    molecule to symmetric adjacency matrix
    """

    m = Chem.Mol(m)

    # m.UpdatePropertyCache()
    # Chem.SetAromaticity(m)
    # if kekulize:
    #     Chem.rdmolops.Kekulize(m)

    ATOM_N = m.GetNumAtoms()
    if MAX_ATOM_N is None:
        MAX_ATOM_N = ATOM_N

    adj = np.zeros((MAX_ATOM_N, MAX_ATOM_N))
    atomic_nums = np.zeros(MAX_ATOM_N)

    assert ATOM_N <= MAX_ATOM_N

    for i in range(ATOM_N):
        a = m.GetAtomWithIdx(i)
        atomic_nums[i] = a.GetAtomicNum()

    for b in m.GetBonds():
        head = b.GetBeginAtomIdx()
        tail = b.GetEndAtomIdx()
        order = b.GetBondTypeAsDouble()
        adj[head, tail] = order
        adj[tail, head] = order
    return atomic_nums, adj



def feat_mol_adj_std(mol,
                 edge_weighted=False, 
                 edge_bin = False,
                 add_identity=False,
                 norm_adj=False, 
                 split_weights = None, 
                 mat_power = 1,
                 use_canon=False):
    """
    Compute the adjacency matrix for this molecule

    If split-weights == [1, 2, 3] then we create separate adj matrices for those
    edge weights

    NOTE: We do not kekulize the molecule, we assume that has already been done

    """
    atomic_nos, adj = mol_to_nums_adj(mol)
    ADJ_N = adj.shape[0]
    input_adj = torch.Tensor(adj)
    
    adj_outs = []

    if edge_weighted:
        adj_weighted = input_adj.unsqueeze(0)
        adj_outs.append(adj_weighted)

    if edge_bin:
        adj_bin = input_adj.unsqueeze(0).clone()
        adj_bin[adj_bin > 0] = 1.0
        adj_outs.append(adj_bin)

    if split_weights is not None:
        split_adj = torch.zeros((len(split_weights), ADJ_N, ADJ_N ))
        for i in range(len(split_weights)):
            split_adj[i] = (input_adj == split_weights[i])
        adj_outs.append(split_adj)

    if use_canon:
        adj_outs.append(torch.tensor(canonical_rank_adj(mol)))

    # if feat_distances:
    #     # print(distances_from_file[molecule_id])
    #     adj_outs += [d.unsqueeze(0) for d in featurize_distance_matrix(distances_from_file[molecule_id], distance_feature_func, bins)]

    adj = torch.cat(adj_outs,0)

    if norm_adj and not add_identity:
        raise ValueError("must add identity if norm adj")
        
    if add_identity:
        adj = adj + torch.eye(ADJ_N) 

    if norm_adj:
        res = []
        for i in range(adj.shape[0]):
            a = adj[i]
            D_12 = 1.0 / torch.sqrt(torch.sum(a, dim=0))

            s1 = D_12.reshape(ADJ_N, 1)
            s2 = D_12.reshape(1, ADJ_N)
            adj_i = s1 * a * s2 

            if isinstance(mat_power, list):
                for p in mat_power:
                    adj_i_pow = torch.matrix_power(adj_i, p)

                    res.append(adj_i_pow)

            else:
                if mat_power > 1: 
                    adj_i = torch.matrix_power(adj_i, mat_power)

                res.append(adj_i)
        adj = torch.stack(res)
    return adj



def whole_molecule_features(full_record, possible_solvents=[], possible_references = []):
    """
    return a vector of features for the full molecule 
    """
    out_feat = []
    if len(possible_solvents) > 0:
        out_feat.append(to_onehot(full_record['solvent'], possible_solvents))

    if len(possible_references) > 0:
        out_feat.append(to_onehot(full_record['reference'], possible_references))

    if len(out_feat) == 0:
        return torch.Tensor([])
    return torch.Tensor(np.concatenate(out_feat).astype(np.float32))

def dist_mat(mol,
             conf_idx = 0,
             feat_distance_pow = [{'pow' : 1,
                                   'max' : 10,
                                   'min' : 0,
                                   'offset' : 0.1}],
             mmff_opt_conf = False, 
             ):
    """
    Return matrix features for molecule
    
    """
    res_mats = []
    if mmff_opt_conf:
        Chem.AllChem.EmbedMolecule(mol)
        Chem.AllChem.MMFFOptimizeMolecule(mol)
    atomic_nos, coords = get_nos_coords(mol, conf_idx)
    ATOM_N = len(atomic_nos)

    pos = coords
    a = pos.T.reshape(1, 3, -1)
    b = np.abs((a - a.T))
    c = np.swapaxes(b, 2, 1)
    c = np.sqrt((c**2).sum(axis=-1))
    dist_mat = torch.Tensor(c).unsqueeze(-1).numpy() # ugh i am sorry    
    for d in feat_distance_pow:
        power  = d.get('pow', 1)
        max_val = d.get('max', 10000)
        min_val = d.get('min', 0)
        offset = d.get('offset', 0)

        v = (dist_mat + offset)**power
        v = np.clip(v, a_min = min_val,
                    a_max = max_val)
        #print("v.shape=", v.shape)
        res_mats.append(v)

        
    if len(res_mats) > 0:
        M = np.concatenate(res_mats, 2)

    assert np.isfinite(M).all()
    return M


def mol_to_nx(mol):
    g = nx.Graph()
    g.add_nodes_from(range(mol.GetNumAtoms()))
    g.add_edges_from([(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), 
     {'weight' : b.GetBondTypeAsDouble()}) for b in mol.GetBonds()])
    return g



w_lut = {1.0 : 0, 1.5 : 1, 2.0: 2, 3.0: 3}

def get_min_path_length(g):
    N = len(g.nodes)
    out = np.zeros((N, N), dtype=np.int32)
    sp = nx.shortest_path(g)
    for i, j in sp.items():
        for jj, path in j.items():
            out[i, jj] = len(path)
    return out

def get_bond_path_counts(g):
    N = len(g.nodes)
    out = np.zeros((N, N, 4), dtype=np.int32)
    sp = nx.shortest_path(g)   
    
    for i, j in sp.items():
        for jj, path in j.items():
            for a, b in zip(path[:-1], path[1:]):
                w = g.edges[a, b]['weight']
                
                out[i, jj, w_lut[w]] +=1
                
    return out

def get_cycle_counts(g, cycle_size_max = 10):
    N = len(g.nodes)
    M = cycle_size_max - 2
    cycle_mat = np.zeros((N, N, M), dtype=np.float32)
    for c in nx.cycle_basis(g):
        x = np.zeros(N)
        x[c] = 1
        if len(c) <= cycle_size_max:
            
            cycle_mat[:, :, len(c)-3] += np.outer(x, x)
    return cycle_mat

def get_graph_props(mol, min_path_length=False,
                    bond_path_counts=False,
                    cycle_counts = False,
                    cycle_size_max=9, 
                    
                    ):
    g = mol_to_nx(mol)

    out = []
    if min_path_length:
        out.append(np.expand_dims(get_min_path_length(g), -1))

    if bond_path_counts:
        out.append(get_bond_path_counts(g))

    if cycle_counts:
        out.append(get_cycle_counts(g, cycle_size_max=cycle_size_max))

    if len(out) == 0:
        return None
    return np.concatenate(out, axis=-1)
        
def get_geom_props(mol,
              dist_mat_mean = False,
              dist_mat_std = False):
    """
    returns geometry features for mol
    
    """
    res_mats = []

    Ds = np.stack([Chem.rdmolops.Get3DDistanceMatrix(mol, c.GetId()) for c in mol.GetConformers()], -1)

    M = None

    if dist_mat_mean:
        D_mean = np.mean(Ds, -1)

        res_mats.append(np.expand_dims(D_mean.astype(np.float32), -1))
        
    if dist_mat_std:
        D_std = np.std(Ds, -1)

        res_mats.append(np.expand_dims(D_std.astype(np.float32), -1))
        
    if len(res_mats) > 0:
        M = np.concatenate(res_mats, 2)


    return M


FEAT_R_GAUSSIAN_FILTERS = {

                           
    'baseline' : [(mu, 0.2) for mu in np.arange(0.5, 6, 0.25)], 
    'dense' :  [(mu, 0.1) for mu in np.arange(0.5, 6, 0.125)], 
    'wide_range' : [(mu, 0.2) for mu in np.arange(0.5, 10, 0.25)], 
    'scaled' :  [(mu, 0.05 * mu ) for mu in np.geomspace(1.5, 10, 30)],
    'scaled_wide' :  [(mu, 0.1 * mu ) for mu in np.geomspace(1.0, 10, 30)],
    'scaled_narrow' :  [(mu, 0.02 * mu ) for mu in np.geomspace(1.0, 10, 50)],
    }
    
    
    
    

                           
def feat_tensor_mol_geom(record,
                         feat_r_pow = None,
                         add_identity=False,
                         feat_r_max = None,
                         feat_r_onehot_tholds = [],
                         feat_r_gaussian_filters = [],
                         feat_angle_gaussian_filters = [],
                         feat_dihedral_gaussian_filters = [],  
                         conf_gauss_bins = False,
                         feat_distances = False,
                         distance_feature_func = 'inv',
                         is_in_ring_size = [],  
                         bins = [2,4,8],
                         MAX_POW_M = 2.0,
                         norm_mat = False):
    """
    All geometry-based atom-pair featurization should go here. 
    
    Note that this takes in the full record and assumes that certain
    featurizations exist already (like 'mean_distance_mat'). 

    Make sure to add on in a way that allows for those featurizations
    to not be present. 
    """
    mol = record['rdmol']
    ATOM_N = mol.GetNumAtoms()
    res_mats = []

    if feat_distances:
        d = record['mean_distance_mat']
        res_mats += [dis.unsqueeze(2) for dis in featurize_distance_matrix(d, distance_feature_func, bins)]

    if conf_gauss_bins:
        b = record['conf_gauss_bins'].transpose((2, 0, 1))
        res_mats += [torch.tensor(bs).unsqueeze(2) for bs in b]

    if feat_r_pow is not None:
        d = record['mean_distance_mat']
        
        e = (np.eye(d.shape[0]) + d)[:, :, np.newaxis]
        if feat_r_max is not None:
            d[d >= feat_r_max] = 0.0
                       
        for p in feat_r_pow:
            e_pow = e**p
            if (e_pow > MAX_POW_M).any():
               # print("WARNING: max(M) = {:3.1f}".format(np.max(e_pow)))
                e_pow = np.minimum(e_pow, MAX_POW_M)

            res_mats.append(e_pow)
        for th in feat_r_onehot_tholds:
            e_oh = (e <= th).astype(np.float32)
            res_mats.append(e_oh)

    if isinstance(feat_r_gaussian_filters, str):
        # perform lookup
        feat_r_gaussian_filters = FEAT_R_GAUSSIAN_FILTERS[feat_r_gaussian_filters]
        
    for mu, sigma in feat_r_gaussian_filters:
        d = record['mean_distance_mat']
        d_val = np.exp(-(d - mu)**2/(2*sigma**2))
        res_mats.append(d_val[:, :, np.newaxis])
            
    for mu, sigma in feat_angle_gaussian_filters:
        d = record['mean_angle_mat']
        d_val = np.exp(-(d - mu)**2/(2*sigma**2))
        res_mats.append(d_val[:, :, np.newaxis])

    for mu, sigma in feat_dihedral_gaussian_filters:
        d = record['mean_angle_dihedral_mat']
        d_val = np.exp(-(d - mu)**2/(2*sigma**2))
        res_mats.append(d_val[:, :, np.newaxis])


    # ring size info

    if is_in_ring_size is not None:
        for rs in is_in_ring_size:
            a = np.zeros((ATOM_N, ATOM_N, 1), dtype=np.float32)
            for b in mol.GetBonds():
                if b.IsInRingSize(rs):
                    a[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] = 1
                    a[b.GetEndAtomIdx(), b.GetBeginAtomIdx()] = 1
            res_mats.append(a)
    
            
    if len(res_mats) > 0:
        M = np.concatenate(res_mats, 2)
    else: # Empty matrix
        M = np.zeros((ATOM_N, ATOM_N, 0), dtype=np.float32)


    M = torch.Tensor(M).permute(2, 0, 1)
    
    if add_identity:
        M = M + torch.eye(ATOM_N).unsqueeze(0)

    if norm_mat:
        res = []
        for i in range(M.shape[0]):
            a = M[i]
            D_12 = 1.0 / torch.sqrt(torch.sum(a, dim=0))
            assert np.min(D_12.numpy()) > 0
            s1 = D_12.reshape(ATOM_N, 1)
            s2 = D_12.reshape(1, ATOM_N)
            adj_i = s1 * a * s2 

            res.append(adj_i)
        M = torch.stack(res, 0)

    #print("M.shape=", M.shape)
    assert np.isfinite(M).all()
    return M.permute(1, 2, 0) 


