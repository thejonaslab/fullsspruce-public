"""
We have identified several places where etkdg does not maintain
atomic identities. These are our attempted workarounds. 

"""

import numpy as np
from rdkit import Chem
import rdkit.Chem.AllChem
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions

def get_all_conf_pos(mol):
    """
    Return np array of  N confs x M atoms x 3 
    conformer positions
    """
    pos = np.stack([c.GetPositions() for c in mol.GetConformers()], axis=0)
    return pos

def copy_confs(conf_source_mol, target_mol):
    """
    Return a new copy of target mol with the 
    conformations from source_mol copied over. 

    A way to transfer confs in the event that the conf-generation
    framework modifies properties of the input molecule
    """
    
    out_mol = Chem.Mol(target_mol)
    out_mol.RemoveAllConformers()
    for c in conf_source_mol.GetConformers():
        out_mol.AddConformer(Chem.Conformer(c))
    return out_mol

def get_double_bonds(mol):
    return [b for b in mol.GetBonds() if b.GetBondTypeAsDouble() == 2.0]


def get_atom_det(p):
    """
    For the 3 atoms in the position matrix, return the determinant. 
    """
    p = np.concatenate([p, np.ones((4, 1))], axis=1)
    return np.linalg.det(p.T)


def get_det_values_for_conf(mol, conf_idx = 0):
    pos = get_all_conf_pos(mol)
    tgt_pos = pos[conf_idx]
    out = {}
    
    for a in mol.GetAtoms():
        n = a.GetNeighbors()
        if len(n) == 4:
            atom_idx = [b.GetIdx() for b in n]
            out[a.GetIdx()] = get_atom_det(tgt_pos[atom_idx])
    return out


def force_stereo_for_double_bonds(mol):
    """
    Force a fixed stereo for double bonds in the mol that do
    not already have stereo set. 
    
    MUTATES THE INPUT MOLECULE
    """
    for b in get_double_bonds(mol):
        if b.GetStereo() == Chem.BondStereo.STEREONONE:
            # FIXME should we have rules here? 
            for begin_stereo in b.GetBeginAtom().GetNeighbors():
                if begin_stereo.GetIdx() != b.GetEndAtom().GetIdx():
                    break
            
            for end_stereo in b.GetEndAtom().GetNeighbors():
                if end_stereo.GetIdx() != b.GetBeginAtom().GetIdx():
                    break
            
            b.SetStereoAtoms(int(begin_stereo.GetIdx()), int(end_stereo.GetIdx()))
            b.SetStereo(Chem.BondStereo.STEREOCIS)
    return None

def force_chiral_for_atoms_with_4_neighbors(mol):
    """
    If a mol has 4 neighbors and does not have chirality set, force it

    MUTATES THE INPUT MOL 
    """
    
    for a in mol.GetAtoms():
        if len(a.GetNeighbors()) == 4:
            if a.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED:
                a.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
    
    return None


def set_chiral_for_all_unset_atoms_based_on_pos(mol, conf_idx=0):
    """
    For all atoms that don't currently have a chirality tag
    set, set it based on the geometry in a way that future runs of 
    etkdg will generate the same chirality

    MUTATES THE INPUT MOLECULE

    """

    dv = get_det_values_for_conf(mol, conf_idx)


    for atom_idx, det_val in dv.items():
        if det_val > 0:
            mol.GetAtomWithIdx(atom_idx).SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
        else:
            mol.GetAtomWithIdx(atom_idx).SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)    

    return None


def set_stereo_for_double_bonds_based_on_pos(mol, conf_idx=0):
    """
    For all bonds that don't currently have a stereochem tag
    set, set it based on the geometry in a way that future runs of 
    etkdg will generate the same geometry

    MUTATES THE INPUT MOLECULE

    """

    dm = Chem.AllChem.Get3DDistanceMatrix(mol, confId = mol.GetConformers()[conf_idx].GetId())


    double_bonds = get_double_bonds(mol)

    for b in double_bonds:
        if not ((b.GetStereo() == Chem.BondStereo.STEREONONE )
                or (b.GetStereo() == Chem.BondStereo.STEREOANY )):
            continue

        
        bond_atom_1 = b.GetBeginAtom()
        bond_atom_2 = b.GetEndAtom()

        if len(bond_atom_1.GetNeighbors()) == 1 or \
           len(bond_atom_2.GetNeighbors()) == 1:
            continue
        
        swapped_12 = False
        
        if len(bond_atom_1.GetNeighbors()) > len(bond_atom_2.GetNeighbors()):
            bond_atom_1, bond_atom_2 = bond_atom_2, bond_atom_1
            swapped_12 = True
        assert len(bond_atom_1.GetNeighbors()) <= len(bond_atom_2.GetNeighbors())

        
        atom_1 = [n1 for n1 in bond_atom_1.GetNeighbors() if
                  n1.GetIdx() != bond_atom_2.GetIdx()][0]
        
        atom_2s = [a for a in bond_atom_2.GetNeighbors() \
                   if a.GetIdx() != bond_atom_1.GetIdx()]

        if len(atom_2s) == 1:
            continue
        
        atom_2_1_d = dm[atom_1.GetIdx(), atom_2s[0].GetIdx() ] 
        atom_2_2_d = dm[atom_1.GetIdx(), atom_2s[1].GetIdx() ]
        if atom_2_1_d < atom_2_2_d :
            atom_2 = atom_2s[0]
        else:
            atom_2 = atom_2s[1]

            
        if swapped_12:
            b.SetStereoAtoms(int(atom_2.GetIdx()), int(atom_1.GetIdx()))
        
        else:
            b.SetStereoAtoms(int(atom_1.GetIdx()), int(atom_2.GetIdx()))
        b.SetStereo(Chem.BondStereo.STEREOCIS)

    return None
    


def generate_clean_etkdg_confs(orig_mol, num=100,
                               assign_chi_to_tet=True,
                               assign_stereo_to_double=True,
                               seed=-1,
                               conform_to_existing_conf_idx=-1,
                               max_embed_attempts=0,
                               exception_for_num_failure=True,
                               num_threads=1):
    """
    Generate etkdg tags in a way that preserve chirality, does the right
    thing with double bonds, etc. 

    Returns a new mol with the generated confs, having removed
    all the old ones. 

    By default ignores existing conformations, but if conform_to_existing_conf_idx
    is set, the "chirality" of the existing conformer will be computed and used. 
    
    max_embed_attempts: etkdg will try this many embeddings before giving up. By default will keep trying. 
    can lead to long embed times

    exception_for_num_failure: raise an exception if we fail to generate num confs

    
    """
    
    
    mol = Chem.Mol(orig_mol)

    if conform_to_existing_conf_idx >= 0:
        assert mol.GetNumConformers() > 0

        set_chiral_for_all_unset_atoms_based_on_pos(mol, conform_to_existing_conf_idx)
        set_stereo_for_double_bonds_based_on_pos(mol, conform_to_existing_conf_idx)

    else:
        if assign_chi_to_tet:
            force_chiral_for_atoms_with_4_neighbors(mol)
        if assign_stereo_to_double:
            force_stereo_for_double_bonds(mol)
        
    
    Chem.AllChem.EmbedMultipleConfs(mol, numConfs=num,
                                    randomSeed=seed,
                                    maxAttempts = max_embed_attempts,
                                    numThreads = num_threads)

    if exception_for_num_failure and mol.GetNumConformers() < num:
        raise Exception(f"Requested {num} conformers but only generated {mol.GetNumConformers()}")

    return copy_confs(mol, orig_mol)

def calc_energies_mmff94(input_mol, conf_ids):
    """
    Calculate the mmff94 energy of conformers in "conf_ids" of input mol. 
    Does not modify the input molecule. 
    """
    mol = Chem.Mol(input_mol)
    props = rdkit.Chem.rdForceFieldHelpers.MMFFGetMoleculeProperties(mol)
    energies = []
    for conf_id in conf_ids:
        ff = rdkit.Chem.rdForceFieldHelpers.MMFFGetMoleculeForceField(mol, props, 
                                                             confId = conf_id)
        
        energies += [ff.CalcEnergy()]

    return energies

def get_viable_stereoisomers(mol, opts = {}, num=-1, max_embed_attempts=0, ignoreFirst=False):
    """
    Return a list of viable stereoisomers of the given mol, using ETKDG directly to determine
    if isomer is physically viable or not rather than tryEmbedding. If num is not 0, cap 
    number of stereoisomers and if not able to generate that many stereoisomers, return all 
    that can be returned.
    """
    opts = StereoEnumerationOptions(**opts)
    isomers = EnumerateStereoisomers(mol, options=opts)
    isomer_l = []
    if ignoreFirst:
        test_m = next(isomers)
    if num == -1:
        num = np.inf
    i = 0
    while i < num:
        found = False
        while not found:
            try:
                test_m = next(isomers)
                generate_clean_etkdg_confs(test_m, 1, max_embed_attempts=max_embed_attempts, seed=0)
            except StopIteration:
                return isomer_l
            except:
                pass
            else:
                found = True
        isomer_l += [test_m]
        i += 1

    return isomer_l