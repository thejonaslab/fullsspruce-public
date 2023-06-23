import numpy as np
from rdkit import Chem
from rdkit.Chem import rdchem
from fullsspruce.featurize.atom_features import to_onehot
from fullsspruce.featurize.molecule_features import mol_to_nums_adj, featurize_distance_matrix

def feat_edges(mol, molecule_id,
                    MAX_ATOM_N=None, 
                    MAX_EDGE_N=None,
                    bond_endpoints=False,
                    is_conjugated=False,
                    is_in_ring=False,
                    stereo=False,
                    feat_distances=False,
                    distances_from_file=None,
                    distance_feature_func='inv',
                    bins=[2,4,8]):
    """
    Create features for edges, edge connectivity
    matrix, and edge/vert matrix

    Note: We really really should parameterize this somehow. 
    """
    atom_n = mol.GetNumAtoms()
    _, vert_adj = mol_to_nums_adj(mol)
    
    double_edge_n =  np.sum(vert_adj > 0)
    assert double_edge_n %2 == 0
    edge_n = double_edge_n // 2
    
    
    edge_adj = np.zeros((edge_n, edge_n))
    edge_vert_adj = np.zeros((edge_n, atom_n))
    
    
    edge_list = []
    for i in range(atom_n):
        for j in range(i +1, atom_n):
            if vert_adj[i, j] > 0:
                edge_list.append((i, j))
                e_idx = len(edge_list) - 1
                edge_vert_adj[e_idx, i] = 1
                edge_vert_adj[e_idx, j] = 1
    # now which edges are connected
    edge_adj = edge_vert_adj @ edge_vert_adj.T - 2*np.eye(edge_n)
    assert edge_adj.shape == (edge_n, edge_n)
    
    # now create edge features
    edge_features = []
    if feat_distances:
        distances = featurize_distance_matrix(distances_from_file[molecule_id], distance_feature_func, bins)
    for b in mol.GetBonds():
        f = []
        f += to_onehot(b.GetBondTypeAsDouble(), [1.0, 1.5, 2.0, 3.0])

        # New features
        beg, end = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if bond_endpoints:
            f += to_onehot(mol.GetAtomWithIdx(beg).GetAtomicNum(), [1, 6, 7, 8, 9])
            f += to_onehot(mol.GetAtomWithIdx(end).GetAtomicNum(), [1, 6, 7, 8, 9])
        if is_conjugated:
            f += [b.GetIsConjugated()]
        if is_in_ring:
            f += [b.IsInRing()]
        if stereo:
            f += to_onehot(b.GetStereo(), list(rdchem.BondStereo.values.values()))
        if feat_distances:
            f += [d[beg, end] for d in distances]

        edge_features.append(f)
    # for edge_idx, (i, j) in enumerate(edge_list):
    #     f = []
    #     f += to_onehot(vert_adj[i, j], [1.0, 1.5, 2.0, 3.0])
        
    #     edge_features.append(f)              
        # maybe do more stuff here? I don't know
        
    edge_features = np.array(edge_features)
    
    return {'edge_edge' : np.expand_dims(edge_adj, 0), 
            'edge_feat' : edge_features, 
            'edge_vert': np.expand_dims(edge_vert_adj, 0)}
