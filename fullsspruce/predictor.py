import numpy as np
import pandas as pd
import torch
import pickle
import os
import time
import collections

from rdkit import Chem

from fullsspruce import predwrap
from fullsspruce import util
from fullsspruce import geom_util

from pathlib import Path

source_path = Path(__file__).resolve()
dir_path = str(source_path.parent)

DEFAULT_MODELS = {
    '1H': {'checkpoint': dir_path + '/default_predict_models/default_1H_model.chk',
            'meta': dir_path + '/default_predict_models/default_1H_model.meta',
            'pred_channels': [1]},
    '13C': {'checkpoint': dir_path + '/default_predict_models/default_13C_model.chk',
            'meta': dir_path + '/default_predict_models/default_13C_model.meta'},
    'coupling': {'checkpoint': dir_path + '/default_predict_models/default_coupling_ETKDG_model.chk',
                    'meta': dir_path + '/default_predict_models/default_coupling_ETKDG_model.meta'}
}

def defaultGeometry(use_confs=False, save_confs=False, prog_bar=False, num_threads=0):
    if use_confs:
        return geom_util.geometryFeaturizerGenerator(['mean_distance_mat', 'mean_angle_mat', 'conf_gauss_bins'], 'confs', 
                            exp_config = {
                                'max_bonds': 4,
                                'mean_mask_choice': 'rdkit',
                                'g_params': geom_util.DEFAULT_GAUSS_BINS
                            },
                            prog_bar=prog_bar)
    else:
        return geom_util.geometryFeaturizerGenerator(['mean_distance_mat', 'mean_angle_mat', 'conf_gauss_bins'], 'ETKDG', 
                            exp_config = {
                                'max_bonds': 4,
                                'mean_mask_choice': 'rdkit',
                                'g_params': geom_util.DEFAULT_GAUSS_BINS,
                                'num_confs': 10,
                                'max_attempts': 20,
                                'optimize': True,
                                'num_threads': num_threads,
                                'return_confs': save_confs
                            },
                            prog_bar=prog_bar)

class Predictor:

    def __init__(self,  models = DEFAULT_MODELS,
                        geometry = {},
                        use_gpu=False,
                        prog_bar=False,
                        batch_size=256,
                        num_threads=0,
                        num_workers=0):
        """
        Defines an NMR property Predictor. Default settings use current best
        models with ML based geometry, and do not require GPU.

        Arguments: 
            models: A dictionary of the checkpoints and meta files for each of the 
                necessary models. 
            geometry: Any object which, when called, takes a list of Mols and returns 
                a list of geometries; or a dictionary specifying which of the default 
                parameters to use (use all default parameters if no argument given).
            use_gpu: Whether to use GPU (if available) 
            prog_bar: Whether to display a progress bar during prediction
            batch_size: Batch size to use during prediction
            num_threads: Number of threads to pass to the ETKDG generator, if applicable.
            num_workers: num_workers to pass to the predictor 
        """
        self.models = models
        self.prog_bar = prog_bar
        if isinstance(geometry, collections.abc.Mapping):
            geom_args = {'prog_bar': self.prog_bar, 'num_threads': num_threads}
            util.recursive_update(geom_args, geometry)
            self.geometry = defaultGeometry(**geom_args)
        else:
            self.geometry = geometry
        self.use_gpu = use_gpu
        self.num_workers = num_workers
        self.BATCH_SIZE = batch_size
        self.metas = {prop: pickle.load(open(model['meta'], 'rb')) for prop,model in self.models.items()}

    def predict(self, mols, 
                    additional_mol_info = None, 
                    properties = ['1H', '13C', 'coupling']):
        """
        Predicts NMR properties for given Mols. 

        Arguments:
            Mols: Either a list of rdkit Mol objects, or a single rdkit Mol object.
            Additional_mol_info: A list of dictionaries to add to each Mol's record for prediction
                (This might include solvent info, geometry info if precomputed, etc.)
            Properties: Which properties to predict for each Mol (default all)
            
        Returns: 
            Predictions: A dictionary or list of dictionaries predicting each mol in Mols
            Meta: A dictionary giving metadata about the process of generating predictions

            **Note: If Mols is a single mol, predict will raise an Exception if there are any
                errors generating predictions. If Mols is a list of mols, then whenever a mol 
                generates an error, its prediction in predictions will be None, and the rest of
                the mols will be predicted as expected. Addional_mol_info, however, should always
                be a list if it is not None.
        """
        # Check for valid mols, only move on with valid ones, and generate geometries
        meta = {}
        m_is_list = isinstance(mols, list)
        mol_l = []
        
        metas = [self.metas[prop] for prop in properties]

        if not m_is_list: # Should be a single Mol
            if additional_mol_info is None:
                additional_mol_info = [{}]
            # Generate/update necessary geometry files
            print("Generating geoms")
            geoms, meta_g = self.geometry([mols])
            util.recursive_update(meta, meta_g)
            start = time.time()
            predictions = [{}]
            orig_idxs = [0]
            valid, err = self.isValidMol(mols, additional_mol_info[0], geoms[0], metas)
            if not valid:
                raise err
            else:
                record = {'rdmol': mols, 'molecule_id': 0, **additional_mol_info[0]}
                util.recursive_update(record, geoms[0])
                mol_l = [record]
        else:
            if additional_mol_info is None:
                additional_mol_info = [{} for _ in mols]
            # Generate/update necessary geometry files
            print("Generating geoms")
            geoms, meta_g = self.geometry(mols)
            util.recursive_update(meta, meta_g)
            start = time.time()
            predictions = [None for _ in mols]
            orig_idxs = []
            mid = 0
            for i, m in enumerate(mols):
                valid, err = self.isValidMol(m, additional_mol_info[i], geoms[i], metas) 
                if valid:
                    record = {'rdmol': m, 'molecule_id': mid, **additional_mol_info[i]}
                    mid += 1
                    util.recursive_update(record, geoms[i])
                    mol_l += [record]
                    predictions[i] = {}
                    orig_idxs += [i]
            if mol_l == []:
                raise ValueError("No valid mols passed.")
        end = time.time()
        meta['validation_time'] = end - start

        # # Write geoms to file if desired
        # if not save_geoms is None:
        #     pickle.dump(geoms, open(save_geoms, 'wb'))

        # Track time for all prediction work
        start = time.time()

        # Create PredModel objects
        if self.use_gpu:
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        if self.use_gpu and not torch.cuda.is_available():
            warnings.warn("CUDA requested but not available, running with CPU")
            device = torch.device('cpu')
        
        predictors = {}
        for prop in properties:
            model_checkpoint_filename, model_meta_filename =  self.models[prop]['checkpoint'], self.models[prop]['meta']
            predictors[prop] = predwrap.PredModel(model_meta_filename, 
                                model_checkpoint_filename, 
                                device,
                                override_pred_config={}, 
            )

        # Make predictions
        for prop, prop_predictor in predictors.items():
            if prop in ['13C', '1H']:
                pred_fields = ['pred_shift_mu', 'pred_shift_std']
            elif prop == 'coupling':
                pred_fields = ['pred_coupling_mu', 'pred_coupling_std']
            else:
                raise ValueError(f"Don't know how to predict {prop}.")
            s = time.time()
            vert_results_df, edge_results_df = prop_predictor.pred(mol_l,
                                                     pred_fields=pred_fields, 
                                                     prog_bar=self.prog_bar,
                                                     BATCH_SIZE=self.BATCH_SIZE,
                                                     num_workers=self.num_workers)
            e = time.time()
            meta['prediction_time_' + prop] = e - s
            # Sort predictions into dictionaries per mol and add the predictions to the list
            # of predictions. Average across predictions from channels in pred_channels (default
            # only use channel 0). 
            if prop in ['13C', '1H']:
                shifts_df = pd.pivot_table(vert_results_df[vert_results_df['pred_chan'].isin(self.models[prop].get('pred_channels', [0]))], 
                    index=['mol_id', 'atom_idx'],
                                columns=['field'],
                                values=['val']).reset_index()
            
                for mol_id, mol_vert_result in shifts_df.groupby('mol_id'):
                    m = mol_l[mol_id]['rdmol']

                    out_shifts = []
                    for row in mol_vert_result.to_dict('records'):
                        atom_idx = int(row[('atom_idx', '')])
                        if m.GetAtomWithIdx(atom_idx).GetAtomicNum() == util.nuc_to_atomicno[prop]:
                            out_shifts.append({'atom_idx' : atom_idx, 
                                            'pred_mu' : row[('val', 'pred_shift_mu')],
                                            'pred_std' : row[('val', 'pred_shift_std')],
                                            })

                    predictions[orig_idxs[mol_id]][prop] = out_shifts 
            elif prop == 'coupling':
                coupling_df = pd.pivot_table(edge_results_df[edge_results_df['pred_chan'].isin(self.models[prop].get('pred_channels', [0]))], 
                        index=['mol_id','atomidx_1','atomidx_2'],
                            columns=['field'],
                            values=['val']).reset_index()
                for mol_id, edge_vert_result in coupling_df.groupby('mol_id'):
                    m = mol_l[mol_id]['rdmol']
                    
                    #tgt_idx = [int(a.GetIdx()) for a in m.GetAtoms() if a.GetAtomicNum() == util.nuc_to_atomicno[to_pred]]

                    #a = mol_vert_result.to_dict('records')
                    out_shifts = []
                    for row in edge_vert_result.to_dict('records'): # mol_vert_result.iterrows():
                        atom_idx1 = int(row[('atomidx_1', '')])
                        atom_idx2 = int(row[('atomidx_2', '')])
                        # if m.GetAtomWithIdx(atom_idx).GetAtomicNum() == util.nuc_to_atomicno[to_pred]:
                        out_shifts.append({'atom_idx1' : atom_idx1, 
                                            'atom_idx2': atom_idx2,
                                            'coup_mu' : row[('val', 'pred_coupling_mu')],
                                            'coup_std' : row[('val', 'pred_coupling_std')],
                                            })
                    predictions[orig_idxs[mol_id]][prop] = out_shifts

        end = time.time()

        meta['prediction_time'] = end - start

        if not m_is_list:
            predictions = predictions[0]

        return predictions, meta

    def isValidMol(self, mol, additional_mol_info, geoms, metas):
        """
        Determine if a mol is in a valid format to be predicted using this Predictor, for the
        models/properties defined by the given metas.

        Arguments:
            Mol: rdkit Mol object
            Additional_mol_info: Additional information on a Mol that may not be contained in the
                Mol object, such as solvent info. 
            Metas: list of metas which Mol must satisfy (dictionaries)

        Returns: 
            Valid: Is the Mol in a valid format.
            Except: Reason that Mol is not valid (or None if it is)
        """
        for meta in metas:
            # Check that all geometry information is present
            geom_args = meta['dataset_hparams']['feat_mol_geom_args']
            try:
                # Check that molecule can be read and processed
                feat_atomicnos = meta['dataset_hparams']['feat_vect_args'].get('feat_atomicno_onehot', [1, 6, 7, 8, 9])
                atomic_nos = util.get_nos(mol)
                if feat_atomicnos is not None:
                    for i in range(mol.GetNumAtoms()):
                        a = mol.GetAtomWithIdx(i)
                        atomic_num = int(atomic_nos[i])
                        if not atomic_num in feat_atomicnos:
                            return False, ValueError("Atom with atomic num " + str(atomic_num) + " is not allowed.")
                if mol.GetNumAtoms() > meta.get('max_n', np.inf):
                    return False, ValueError('Too many atoms.')
                if meta['dataset_hparams']['feat_vect_args'].get('rad_electrons', False) and mol.GetNumRadicalElectrons() > 0:
                    return False, ValueError('RADICAL')
                # Check that all geometry and feature information is present
                if not isinstance(additional_mol_info, collections.abc.Mapping):
                    return False, ValueError("Additional Mol info cannot be mapped to record.")
                if not isinstance(geoms, collections.abc.Mapping):
                    return False, ValueError('Geoms cannot be mapped to record.')
                if ('feat_distances' in geom_args or 'feat_r_pow' in geom_args or 'feat_r_gaussian_filters' in geom_args) and not ('mean_distance_mat' in additional_mol_info or 'mean_distance_mat' in geoms):
                    return False, ValueError('Mean Distance Matrix Missing')
                if ('feat_angle_gaussian_filters' in geom_args) and not ('mean_angle_mat' in additional_mol_info or 'mean_angle_mat' in geoms):
                    return False, ValueError("Mean Angle Matrix Missing")
                if ('conf_gauss_bins' in geom_args) and not ('conf_gauss_bins' in additional_mol_info or 'conf_gauss_bins' in geoms):
                    return False, ValueError("Gauss Bins Missing")
                if ('feat_dihedral_gaussian_filters' in geom_args) and not ('mean_angle_dihedral_mat' in additional_mol_info or 'mean_angle_dihedral_mat' in geoms):
                    return False, ValueError("Mean Angle Matrix Missing")
                # Check for miscellaneous other required properties
                if 'possible_solvents' in meta['dataset_hparams']['mol_args'] and not 'solvent' in additional_mol_info:
                    return False, ValueError("Solvent information missing.")
                if 'possible_references' in meta['dataset_hparams']['mol_args'] and not 'reference' in additional_mol_info:
                    return False, ValueError("NMR Reference information missing.")
            except Exception as e:
                return False, e
        return True, None

    def is_valid_mol(self, mol, properties=None):
        """
        Determine if a mol is in a valid format to be predicted using this Predictor, prior
        to any ETKDG embedding or other advanced processing. 

        Arguments:
            Mol: rdkit Mol object

        Returns: 
            Valid: Is the Mol in a valid format.
            Reason: String describing reason that Mol is not valid (or None if it is)
        """
        # Check that molecule can be read and processed
        validity_markers = {}
        validity_markers['feat_atomicnos'] = set(range(100))
        for _, meta in self.metas.items():
            validity_markers['feat_atomicnos'] = validity_markers['feat_atomicnos'] & set(meta['dataset_hparams']['feat_vect_args'].get('feat_atomicno_onehot', [1, 6, 7, 8, 9]))
            validity_markers['max_n'] = min(validity_markers.get('max_n', np.inf), meta.get('max_n', np.inf))
            validity_markers['rad_electrons'] = validity_markers.get('rad_electrons', False) or meta['dataset_hparams']['feat_vect_args'].get('rad_electrons', False)
        atomic_nos = util.get_nos(mol)
        if validity_markers['feat_atomicnos'] is not None:
            for i in range(mol.GetNumAtoms()):
                a = mol.GetAtomWithIdx(i)
                atomic_num = int(atomic_nos[i])
                if not atomic_num in validity_markers['feat_atomicnos']:
                    return False, "Atom with atomic num " + str(atomic_num) + " is not allowed."
        if mol.GetNumAtoms() > validity_markers.get('max_n', np.inf):
            return False, 'Too many atoms.'
        if validity_markers.get('rad_electrons', False) and mol.GetNumRadicalElectrons() > 0:
            return False, 'Molecule illegally contains radicals.'
        return True, None

