"""
Code to wrap the model such that we can easily use
it as a predictor. The goal is to move as much 
model-specific code out of the main codepath. 

"""

import pickle
from itertools import combinations

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from fullsspruce.featurize.netdataio import MoleculeDatasetMulti
from fullsspruce import netutil 
import pandas as pd
from fullsspruce import util
import copy

class PredModel(object):
    """
    Predictor can predict two types of values, 
    per-vert and per-edge. 

    """
    
    def __init__(self, meta_filename, checkpoint_filename,
                 device, override_pred_config=None):

        meta = pickle.load(open(meta_filename, 'rb'))
        self.meta = meta 

        self.device=device
        net = torch.load(checkpoint_filename, map_location=device)
        self.net = net
        self.net.eval()
        self.override_pred_config = override_pred_config

    def pred(self, records, BATCH_SIZE = 32, 
             debug=False, prog_bar= False,
             pred_fields = None, return_res = False, num_workers=0):

        dataset_hparams = copy.deepcopy(netutil.DEFAULT_DATA_HPARAMS)
        util.recursive_update(dataset_hparams, 
                          self.meta['dataset_hparams'])
        # dataset_hparams = self.meta['dataset_hparams']
        # print(dataset_hparams)
        MAX_N = self.meta.get('max_n', 32)
        dataset_class = self.meta.get('ds_class', 'MoleculeDatasetMulti')

        d_c = eval(dataset_class)

        other_args = dataset_hparams.get('other_args', {})

        #pred_config = self.meta.get('pred_config', {})
        #passthrough_config = self.meta.get('passthrough_config', {})

        ### pred-config controls the extraction of true values for supervised
        ### training and is generally not used at pure-prediction time
        if self.override_pred_config is not None:
            pred_config = self.override_pred_config
        else:
            pred_config = self.meta['pred_config']
        passthrough_config = self.meta['passthrough_config']

        # we force set this here
        if 'allow_cache' in other_args:
            del other_args['allow_cache']

        # if not metafile is None:
        #     records_metafile = pickle.load(open(metafile, 'rb'))
        #     records = [r for r in records if r['molecule_id'] in records_metafile['ids']]

        ds = d_c(records, 
                    MAX_N,
                    # metafile,
                    dataset_hparams['feat_vect_args'],
                    dataset_hparams['feat_edge_args'],
                    dataset_hparams['feat_mol_geom_args'],
                    dataset_hparams['feat_bond_args'],
                    dataset_hparams['adj_args'],
                    dataset_hparams['mol_args'],
                    dataset_hparams['dist_mat_args'],
                    dataset_hparams['coupling_args'],
                    pred_config = pred_config,
                    passthrough_config = passthrough_config, 
                    #combine_mat_vect=COMBINE_MAT_VECT,
                    allow_cache=False, **other_args)   
        dl = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                                         num_workers=num_workers)

        # if not metafile is None:
        #     records_metafile = pickle.load(open(metafile, 'rb'))
        #     records = [r for r in records if r['molecule_id'] in records_metafile['ids']]

        results_df = []
        
        res = netutil.run_epoch(self.net, self.device, None, None, dl, 
                                pred_only = True, 
                                return_pred = True, print_shapes=debug, desc='predict',
                                progress_bar=prog_bar)
        
        if return_res:
            return res # debug
        # by default we predict everything the net throws at us
        if pred_fields is None:
            pred_fields = [f for f in list(res.keys()) if f.startswith("pred_")]

        for f in pred_fields:
            if f not in res:
                raise Exception(f"{f} not in res, {list(res.keys())}")

        per_vert_fields = []
        per_edge_fields = []
        for field in pred_fields:
            if 'shift' in field:
                per_vert_fields.append(field)
            else:
                per_edge_fields.append(field)

        ### create the per-vertex fields
        per_vert_out = []
        for rec_i, rec in enumerate(records):
            rdmol = rec['rdmol']
            mol_id = rec['molecule_id']
            atom_n = rdmol.GetNumAtoms()
            
            for atom_idx in range(atom_n):
                vert_rec = {'rec_idx' : rec_i,
                            'mol_id': mol_id,
                            'atom_idx' : atom_idx}
                for field in per_vert_fields:
                    for ji, v in enumerate(res[field][rec_i, atom_idx]):
                        vr = vert_rec.copy()
                        vr['val'] = v
                        vr['field'] = field
                        vr['pred_chan'] = ji
                        per_vert_out.append(vr)
        
        vert_results_df = pd.DataFrame(per_vert_out)

        ### create the per-edge fields
        if len(per_edge_fields) == 0:
            edge_results_df = None
        else:

            per_edge_out = []
            for rec_i, rec in enumerate(records):
                rdmol = rec['rdmol']
                mol_id = rec['molecule_id']
                atom_n = rdmol.GetNumAtoms()

                # for atomidx_1 in range(atom_n):
                #     for atomidx_2 in range(atomidx_1 +1, atom_n):
                for (atomidx_1, atomidx_2) in combinations(range(atom_n), 2):
                    # print(res['pred_passthrough_coupling_types_encoded'][rec_i, 17:25, 17:25])
                    if res['pred_passthrough_coupling_types_encoded'][rec_i, atomidx_1, atomidx_2] == -1:
                        continue
                    edge_rec = {'rec_idx' : rec_i,
                                'mol_id': mol_id,
                                'atomidx_1' : atomidx_1,
                                'atomidx_2' : atomidx_2}

                    for field in per_edge_fields:
                        for ji, v in enumerate(res[field][rec_i, atomidx_1, atomidx_2]):
                            er = edge_rec.copy()
                            er['val'] = v
                            er['field'] = field
                            er['pred_chan'] = ji
                            per_edge_out.append(er)
                        
            edge_results_df = pd.DataFrame(per_edge_out)
            #edge_results_df['atomidx_1'] = edge_results_df['atomidx_1'].astype(int)
            #edge_results_df['atomidx_2'] = edge_results_df['atomidx_2'].astype(int)



        return vert_results_df, edge_results_df

        

if __name__ == "__main__":
    pass

