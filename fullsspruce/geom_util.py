import numpy as np
from tqdm import tqdm
from scipy.stats import norm
import scipy

import tinygraph.io.rdkit
import tinygraph as tg

import rdkit
from rdkit import Chem
from rdkit.Chem.rdDistGeom import GetMoleculeBoundsMatrix
import time

from fullsspruce import util
import warnings

from fullsspruce import etkdg

# try:
#     from vonmises.predictor import *
#     # from vonmises import etkdg
# except:
#     warnings.warn("Could not import vonmises, some functions won't work")



DEFAULT_GAUSS_BINS = [(mu, 0.2) for mu in np.linspace(0.3, 10.0, 20)]

best_VM_model = '/data/williamsjl/vonmises/models/basic-nmrshiftdb/basic-nmrshiftdb.nmrshiftdb-pt-conf-mols.356435903323'

def get_mask_values(mol, mask_choice):
    if mask_choice == 'rdkit':
        return getBoundsMatrixCorrected(mol)
    elif mask_choice == 'mean_default':
        return np.inf
    else:
        return 0

def getBoundsMatrixCorrected(mol):
    """
    rdkit's bound matrix uses 1000 as a magic number indicating an unknown large
    distance. We want to replace this with some reasonable large upper bound. To 
    that end, here we replace it with the largest non-1000 distance plus 2 times the 
    smallest non-zero distance. 
    """
    bounds_matrix = GetMoleculeBoundsMatrix(mol)
    far = np.max(np.where(bounds_matrix < 1000, bounds_matrix, 0))
    close = np.min(np.where(bounds_matrix > 0, bounds_matrix, 1000))
    bounds_matrix[bounds_matrix == 1000] = far + (2*close)
    return (bounds_matrix + bounds_matrix.T)/2  

class GeometryGenerator:
    """
    Generic class which acts as a function returning geometry information
    for a list of mols.
    """
    def __init__(self, geoms, exp_config, prog_bar):
        """
        Class which generates functions to get geometry information from 
        mols using embedded conformers in the RDKit objects.

        Arguments:
            geoms: List of geometry features to generate (names as strings)
            exp_config: Additional configuration arguments
            prog_bar: Whether to display progress bar (tqdm)
        """
        self.geoms = geoms
        self.exp_config = exp_config
        self.prog_bar = prog_bar

    def __call__(self, mols):
        """
        Function which determines geometry information from a list of mols

        Arguments: 
            mols: A list of RDKit Mol objects
        
        Returns:
            features: List of dictionaries, where each dictionary maps from 
                a geom entry to the feature for a particular Mol. 
            meta: Dictionary of meta information from generating the geometries.
        """
        return [{} for m in mols], {}

class ConformerGeometryGenerator(GeometryGenerator):
    """
    Class which generates functions to get geometry information from 
    mols using embedded conformers in the RDKit objects.

    Additional configuration arguments:
        max_bonds: If present, max distance between atoms to generate distances
        mean_mask_choice: If max_bonds, how to assign distances beyond max_bonds
        g_params: If using conf_gauss_bins, parameterization of the bins
        gauss_max_chocie: If max_bonds, how to assign distances beyond max_bonds when generating gauss bins
        boltzmann_weight: If True, get the energy of each conformer using MMFF94 and weight the features
            according to the Boltzmann probability of the respective conformers
    """

    def __call__(self, mols,conf_idxs=None):
        features, meta = [], {}
        start = time.time()

        # mol_energies = self.exp_config.get('energies', [None for _ in mols])

        for i, mol in enumerate(tqdm(mols, desc='Generating features', disable=not self.prog_bar)):
            features += [self.get_features(mol, conf_idxs)]

        end = time.time()
        meta['geoms_from_confs_time'] = end - start
        return features, meta

    def get_features(self, mol, conf_idxs):
        try:
            probabilities = None
            if self.exp_config.get('boltzmann_weight', False):
                e_kcalmol = etkdg.calc_energies_mmff94(mol, util.assert_conf_ID_uniqueness(mol, conf_idxs=conf_idxs))
                probabilities = util.compute_conf_probs(np.array(e_kcalmol)/util.EV_TO_KCAL_MOL)
            mol_f = {}
            for geom in self.geoms:
                if geom == 'mean_distance_mat':
                    f = np.average(util.get_stacked_distance_mat(mol,conf_idxs=conf_idxs), weights=probabilities, axis=-1)
                    if 'max_bonds' in self.exp_config:
                        mb = self.exp_config['max_bonds']
                        tg_mol = tinygraph.io.rdkit.from_rdkit_mol(mol)
                        sp = tg.algorithms.get_shortest_paths(tg_mol, False)
                        mask_out = (sp > mb)
                        mask_vals = get_mask_values(mol, self.exp_config.get('mean_mask_choice', 'mean_default'))
                        f = np.where(mask_out, mask_vals, f)
                elif geom == "conf_gauss_bins":
                    # gaussian-encoded histogram of distances from PT data
                    gaussians = self.exp_config['g_params']
                    e_vals = []

                    dists = util.get_stacked_distance_mat(mol,conf_idxs=conf_idxs)
                    
                    mask_out = np.full_like(dists, False, dtype=np.int64)
                    mask_values = get_mask_values(mol, self.exp_config.get('gauss_mask_choice', 'gauss_default'))
                    if 'max_bonds' in self.exp_config:
                        mb = self.exp_config['max_bonds']
                        tg_mol = tinygraph.io.rdkit.from_rdkit_mol(mol)
                        sp = tg.algorithms.get_shortest_paths(tg_mol, False)
                        mask_out = np.stack([(sp > mb) for _ in range(dists.shape[2])], axis=2)
                    for mu, sigma in gaussians:
                        gauss = np.exp(-(dists - mu)**2/(2*sigma**2))
                        e_val = np.where(mask_out, mask_values, gauss)
                        e_vals.append(np.average(e_val, weights=probabilities, axis=-1))
                    f = np.stack(e_vals, -1).astype(np.float32)
                elif geom == "mean_angle_mat":
                    f = np.average(util.get_conf_angles(mol,conf_idxs=conf_idxs), weights=probabilities, axis=-1)
                elif geom == 'mean_dihedral_angle_mat':
                    f = scipy.stats.circmean(util.get_conf_dihedral_angles(mol,conf_idxs=conf_idxs), low=-np.pi, high=np.pi,axis=-1)
                    # f = np.mean(util.get_conf_dihedral_angles(mol), axis=-1)
                else: 
                    raise Exception("Unrecognized geometry feature.")
                mol_f[geom] = f
        except Exception as e:
            return None
        else:
            return mol_f

# class VMGeometryGenerator(GeometryGenerator):
#     """
#     Class which generates functions to get geometry information from 
#     mols using VM predictions and functions.

#     Additional configuration arguments
#         model: Path to VM model to use for prediction
#         max_bonds: If present, max distance between atoms to generate distances
#         mean_mask_choice: If max_bonds, how to assign distances beyond max_bonds
    # """

    # def __call__(self, mols):
    #     features, meta = [], {}

    #     # Temp fix for compute_torsions issue
    #     prep_mols = [etkdg.generate_clean_etkdg_confs(orig_mol, num=1, seed = 0, max_embed_attempts=5) for orig_mol in mols]

    #     vm_predictor = Predictor(self.exp_config['model'] + '.best.chk', self.exp_config['model'] + '.yaml', use_cuda=True)
    #     preds, meta = vm_predictor.predict(prep_mols)
        
    #     for i, p in enumerate(tqdm(preds, desc='Generating features', disable=not self.prog_bar)):
    #         features += [self.get_features(p)]

    #     return features, meta

    # def get_features(self, p):
    #     # TODO: How to handle one of the preds is None? Not sure, but it will probably raise an 
    #     # Exception right now. Maybe this could be avoid in isValid as well. 
    #     mol_f = {}
    #     for geom in self.geoms:
    #         if geom == 'mean_distance_mat':
    #             vm_f = compute_average_distances_up_to_max_path_len(p, self.exp_config['max_bonds'] + 1, use_cuda=True)
    #             mask_out = (vm_d == np.inf)
    #             mask_vals = get_mask_values(mol, self.exp_config['f_config'].get('mean_mask_choice', 'gauss_default'))
    #             f = np.where(mask_out, mask_vals, means)
    #         elif geom == 'mean_angle_mat':
    #             angles_d = compute_average_angles(pred)
    #             N = Chem.GetNumAtoms(mol)
    #             f = np.zeros((N, N))
    #             for (i, k, j), ang in angles_d.items():
    #                 # i, j = a1.GetIdx(), a2.GetIdx()
    #                 i, j = min(i, j), max(i, j) 
    #                 f[i,j] = ang
    #         # elif geom == 'conf_gauss_bins':
    #             # Not sure how to do this yet
    #         else: 
    #             raise Exception("Unrecognized geometry feature.")
    #         mol_f[geom] = f
    #     return mol_f


class ETKDGGeometryGenerator(GeometryGenerator):
    """
    Take in a list of mols and return the desired geometry features using
    RDKit and ETKDG functions on the base Mol.

    Additional configuration arguments
            num_confs: Number of conformers to generate from etkdg
            seed: Random seed to use for conf generation (default 0)
            max_attempts: Maximum attempts at which to stop generating conformers for etkdg
            max_bonds: If present, max distance between atoms to generate distances
            mean_mask_choice: If max_bonds, how to assign distances beyond max_bonds
            optimize: Whether to perform MMFF optimization on generated confs (default False)
            num_threads: The number of threads to use (If set to zero, the max supported by the system 
                will be used [default])
            return_confs: If True, add the list of Mols with their associated confs added to the meta
    """

    def __call__(self, mols):
        start = time.time()
        mols_with_etkdg_confs = []

        for m in tqdm(mols, desc='Generating ETKDG conformers', disable=not self.prog_bar):
            new_m = self.create_ETKDG_confs(m)
            mols_with_etkdg_confs += [new_m]
        end = time.time()
        gen = ConformerGeometryGenerator(self.geoms, self.exp_config, self.prog_bar)
        features, meta = gen(mols_with_etkdg_confs)
        meta['etkdg_embed_time'] = end - start
        if self.exp_config.get('return_confs', False):
            meta['mols_with_confs'] = mols_with_etkdg_confs
        return features, meta

    def create_ETKDG_confs(self, m):
        try: 
            m_etkdg = etkdg.generate_clean_etkdg_confs(m, num=self.exp_config['num_confs'], 
                                                          seed = self.exp_config.get('seed', 1), 
                                                          max_embed_attempts=self.exp_config['max_attempts'], 
                                                          num_threads=self.exp_config.get('num_threads', 0))
            if self.exp_config.get('optimize', False):
                Chem.AllChem.MMFFOptimizeMoleculeConfs(m_etkdg, numThreads=self.exp_config.get('num_threads', 0))
        except Exception as e:
            return None
        else:
            return m_etkdg

class EmptyGeomGenerator(GeometryGenerator):
    """
    Geometry Generator which generates empty dictionaries for each Mol
    """
    
    def __call__(self, mols):
        return [{} for _ in mols], {}


def geometryFeaturizerGenerator(geoms, func, exp_config = {}, prog_bar=False):
    """
    This is a Generator class designed to create a function that is passed
    to the Predictor class' geometry property.

    Arguments:
        geoms: List of geometry features to generate (names as strings)
        func: Which function in the generator to use (name as string)
        exp_config: Any additional information that will needed to generate
            the features.

    Returns: 
        geometryFeaturizer: A GeometryGenerator which takes in a list of mols and
            returns the desired geometry features.
    """ 
    # if func == "VM":
    #     return VMGeometryGenerator(geoms, exp_config, prog_bar)
    if func == "confs":
        return ConformerGeometryGenerator(geoms, exp_config, prog_bar)
    elif func == 'ETKDG':
        return ETKDGGeometryGenerator(geoms, exp_config, prog_bar)
    elif func == 'empty':
        return EmptyGeomGenerator(geoms, exp_config, prog_bar)
    else:
        raise Exception(f"Unknown featurization function {func}.")
